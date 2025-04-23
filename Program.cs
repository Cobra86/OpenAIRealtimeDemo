using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using NAudio.Wave;
using OpenAI.RealtimeConversation;
using System.ClientModel;
using Azure.AI.OpenAI;
using System.Text; // Required for Azure OpenAI

namespace OpenAIRealtimeDemo
{
#pragma warning disable OPENAI002 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    internal class Program
    {
        // Audio constants
        const int SR = 24_000;   // sample‑rate Hz
        const int BPS = 16;      // bits per sample
        const int CH = 1;        // mono PCM

        // Session state tracking
        static bool isPlayingAudio = false;
        static bool manualModeActive = true;
        static WaveInEvent? microphoneInstance = null;
        static bool microphoneMuted = false;

        // Audio response handling
        static List<byte[]> audioResponseBuffer = new List<byte[]>();
        static bool collectingAudioResponse = false;
        static DateTime lastAudioOutput = DateTime.MinValue;
        static readonly TimeSpan audioBufferTime = TimeSpan.FromMilliseconds(500); // Buffer time after audio

        // Speaker output for handling audio playback
        static SpeakerOutput? speakerOutput = null;

        public static async Task Main(string[] args)
        {
            // Load configuration
            var config = new ConfigurationBuilder()
                .AddUserSecrets<Program>()                
                .Build();

            // Determine which API to use (OpenAI or Azure OpenAI)
            bool useAzure = false;
            string? useAzureStr = config["UseAzure"];
            if (!string.IsNullOrWhiteSpace(useAzureStr) && bool.TryParse(useAzureStr, out bool result))
            {
                useAzure = result;
            }

            Console.WriteLine($"Using API: {(useAzure ? "Azure OpenAI" : "OpenAI")}");

            // Initialize the appropriate client
            RealtimeConversationClient realtimeClient;

            if (useAzure)
            {
                string? endpoint = config["AzureOpenAI:Endpoint"];
                string? apiKey = config["AzureOpenAI:ApiKey"];
                string? deploymentName = config["AzureOpenAI:DeploymentName"];

                if (string.IsNullOrWhiteSpace(endpoint) || string.IsNullOrWhiteSpace(apiKey) || string.IsNullOrWhiteSpace(deploymentName))
                {
                    Console.WriteLine("Azure OpenAI configuration is incomplete. Please set AzureOpenAI:Endpoint, AzureOpenAI:ApiKey, and AzureOpenAI:DeploymentName in configuration.");
                    return;
                }

                // Initialize Azure OpenAI client
                realtimeClient = Helpers.GetAzureRealtimeConversationClient(endpoint, apiKey, deploymentName);
                Console.WriteLine($"Connected to Azure OpenAI endpoint: {endpoint}");
                Console.WriteLine($"Using deployment: {deploymentName}");
            }
            else
            {
                string? apiKey = config["OpenAIKey"];

                if (string.IsNullOrWhiteSpace(apiKey))
                {
                    Console.WriteLine("OpenAI API key not found. Please set OpenAIKey in configuration or environment variable.");
                    return;
                }

                // Initialize OpenAI client
                realtimeClient = Helpers.GetRealtimeConversationClient(apiKey);
                Console.WriteLine("Connected to OpenAI API");
            }

            // Build kernel with plugins
            var kernel = Kernel.CreateBuilder().Build();
            kernel.ImportPluginFromType<WeatherPlugin>();

            /* Select microphone device */
            int deviceCount = WaveInEvent.DeviceCount;
            Console.WriteLine("\nAvailable microphones:");
            for (int i = 0; i < deviceCount; i++)
                Console.WriteLine($"{i}: {WaveInEvent.GetCapabilities(i).ProductName}");

            Console.Write("Choose device #: ");
            if (!int.TryParse(Console.ReadLine(), out int dev) ||
                dev < 0 || dev >= deviceCount) return;

            /* Choose mode */
            Console.WriteLine("\nChoose operation mode:");
            Console.WriteLine("1: Auto mode (automatic voice detection)");
            Console.WriteLine("2: Manual mode (press ENTER to get response)");
            Console.Write("Enter mode (1 or 2): ");

            if (!int.TryParse(Console.ReadLine(), out int mode) || (mode != 1 && mode != 2))
            {
                Console.WriteLine("Invalid mode selection. Defaulting to Manual mode.");
                mode = 2;
            }

            manualModeActive = (mode == 2);

            try
            {
                // Initialize speaker output
                speakerOutput = new SpeakerOutput();

                // Start a new conversation session - Create session outside using to manage lifecycle better
                RealtimeConversationSession session = await realtimeClient.StartConversationSessionAsync();

                // Flag to track session state
                bool sessionActive = true;

                try // Nested try block to ensure proper session disposal
                {
                    // Configure the session based on selected mode
                    var sessionOptions = new ConversationSessionOptions
                    {
                        Voice = ConversationVoice.Alloy,
                        InputAudioFormat = ConversationAudioFormat.Pcm16,
                        OutputAudioFormat = ConversationAudioFormat.Pcm16,
                        InputTranscriptionOptions = new()
                        {
                            Model = "whisper-1",
                        },
                    };

                    // Add plugins from kernel as tools
                    foreach (var tool in Helpers.ConvertFunctions(kernel))
                    {
                        sessionOptions.Tools.Add(tool);
                    }

                    // If using auto mode, configure turn detection
                    if (!manualModeActive)
                    {
                        sessionOptions.TurnDetectionOptions = ConversationTurnDetectionOptions.CreateServerVoiceActivityTurnDetectionOptions(
                            detectionThreshold: 0.6f,
                            prefixPaddingDuration: TimeSpan.FromMilliseconds(300),
                            silenceDuration: TimeSpan.FromMilliseconds(700),
                            enableAutomaticResponseCreation: true);
                        Console.WriteLine("Auto mode enabled with smooth settings");
                    }
                    else
                    {
                        sessionOptions.TurnDetectionOptions = ConversationTurnDetectionOptions.CreateDisabledTurnDetectionOptions();
                        Console.WriteLine("Manual mode enabled: Press ENTER to request a response");
                    }

                    // If tools are available, enable auto tool selection
                    if (sessionOptions.Tools.Count > 0)
                    {
                        sessionOptions.ToolChoice = ConversationToolChoice.CreateAutoToolChoice();
                    }

                    // Configure the session
                    await session.ConfigureSessionAsync(sessionOptions);
                    Console.WriteLine("Session configured successfully");

                    // Initialize the microphone
                    microphoneInstance = new WaveInEvent
                    {
                        DeviceNumber = dev,
                        WaveFormat = new WaveFormat(SR, BPS, CH),
                        BufferMilliseconds = 100
                    };

                    // Handle incoming audio from microphone
                    microphoneInstance.DataAvailable += async (_, e) =>
                    {
                        try
                        {
                            // Skip sending audio if in manual mode and playing audio, or if mic is muted or session inactive
                            if ((manualModeActive && isPlayingAudio) || microphoneMuted || !sessionActive) return;

                            // Create a copy of the audio data for analysis
                            var pcm = new byte[e.BytesRecorded];
                            Array.Copy(e.Buffer, pcm, pcm.Length);

                            // In auto mode when AI is speaking, check if this is from the AI before sending
                            if (isPlayingAudio && !manualModeActive)
                            {
                                // Check if the audio is likely from the AI output
                                if (AudioProcessor.IsLikelyAIVoice(pcm))
                                {
                                    // Skip this audio as it's likely AI feedback
                                    return;
                                }

                                // If we're here, it might be a human interruption - log it
                                Console.WriteLine("Possible human interruption detected");

                                // Clear speaker output to prevent feedback when the user is speaking
                                speakerOutput?.ClearPlayback();
                            }
                            // General time-based prevention (additional safety)
                            else if (DateTime.Now - lastAudioOutput < audioBufferTime)
                            {
                                return;
                            }

                            try
                            {
                                // Send raw audio data to the session
                                await session.SendInputAudioAsync(new MemoryStream(pcm));
                            }
                            catch (ObjectDisposedException)
                            {
                                // Gracefully handle the case where the session is already disposed
                                sessionActive = false;
                                Console.WriteLine("Session connection closed. Please restart the application.");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error sending audio: {ex.Message}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error in microphone data handler: {ex.Message}");
                        }
                    };

                    // Start the microphone
                    Console.WriteLine("Starting recording...");
                    microphoneInstance.StartRecording();

                    // Start the message receiver task for processing server updates
                    var cts = new CancellationTokenSource();
                    _ = Task.Run(() => ProcessSessionUpdatesAsync(session, kernel, cts.Token, () => sessionActive = false));

                    // Command loop
                    Console.WriteLine("\n=== COMMANDS ===");
                    if (manualModeActive)
                    {
                        Console.WriteLine(">> Press ENTER: Get AI response");
                    }
                    else
                    {
                        Console.WriteLine(">> Auto mode is active - just speak when the AI is not speaking");
                    }
                    Console.WriteLine(">> Type 'quit': Exit the program");
                    Console.WriteLine("===============");

                    bool running = true;
                    while (running)
                    {
                        string? cmd = Console.ReadLine()?.Trim().ToLower();

                        if (cmd == "quit")
                        {
                            running = false;
                        }
                        else if (manualModeActive && string.IsNullOrEmpty(cmd) && sessionActive)
                        {
                            // In manual mode, ENTER triggers response generation
                            try
                            {
                                // Temporarily mute microphone during processing
                                microphoneMuted = true;
                                if (microphoneInstance != null)
                                {
                                    microphoneInstance.StopRecording();
                                    Console.WriteLine("Microphone paused while processing response");
                                }

                                // Clear any previous audio buffer
                                audioResponseBuffer.Clear();
                                collectingAudioResponse = true;

                                // Request response generation
                                Console.WriteLine("Requesting AI response...");
                                await session.StartResponseAsync();
                            }
                            catch (ObjectDisposedException)
                            {
                                Console.WriteLine("Session connection closed. Please restart the application.");
                                sessionActive = false;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error requesting response: {ex.Message}");
                                microphoneMuted = false;

                                // Resume microphone
                                if (microphoneInstance != null && !isPlayingAudio)
                                {
                                    try
                                    {
                                        microphoneInstance.StartRecording();
                                    }
                                    catch (InvalidOperationException)
                                    {
                                        Console.WriteLine("Microphone already recording");
                                    }
                                }
                            }
                        }
                    }

                    // Clean up
                    Console.WriteLine("Shutting down...");
                    cts.Cancel();
                }
                finally
                {
                    // Ensure resources are always properly disposed
                    if (microphoneInstance != null)
                    {
                        try
                        {
                            microphoneInstance.StopRecording();
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error stopping microphone: {ex.Message}");
                        }
                        microphoneInstance.Dispose();
                        microphoneInstance = null;
                    }

                    speakerOutput?.Dispose();
                    speakerOutput = null;

                    session.Dispose();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }

        // Helper method to process session updates
        private static async Task ProcessSessionUpdatesAsync(RealtimeConversationSession session, Kernel kernel,
                                              CancellationToken ct, Action onSessionClosed = null)
        {
            string textResponse = "";
            bool isFirstAudioInResponse = true;
            Dictionary<string, StringBuilder> functionArgumentBuildersById = [];

            try
            {
                await foreach (ConversationUpdate update in session.ReceiveUpdatesAsync(ct))
                {
                    // Handle different update types
                    if (update is ConversationSessionStartedUpdate sessionStartedUpdate)
                    {
                        Console.WriteLine($"<<< Session started. ID: {sessionStartedUpdate.SessionId}");
                    }
                    else if (update is ConversationInputSpeechStartedUpdate)
                    {
                        Console.WriteLine("  -- Voice activity detection started");

                        // Clear any playing audio when user starts speaking to prevent feedback
                        speakerOutput?.ClearPlayback();
                    }
                    else if (update is ConversationInputSpeechFinishedUpdate)
                    {
                        Console.WriteLine("  -- Voice activity detection ended");
                    }
                    else if (update is ConversationItemStreamingStartedUpdate itemStreamingStartedUpdate)
                    {
                        isFirstAudioInResponse = true;
                        textResponse = "";

                        Console.WriteLine("  -- Begin streaming of new item");
                        if (!string.IsNullOrEmpty(itemStreamingStartedUpdate.FunctionName))
                        {
                            Console.WriteLine($"    {itemStreamingStartedUpdate.FunctionName}: ");
                        }
                    }
                    else if (update is ConversationItemStreamingPartDeltaUpdate deltaUpdate)
                    {
                        // Handle transcription or text
                        if (!string.IsNullOrEmpty(deltaUpdate.AudioTranscript))
                        {
                            Console.Write(deltaUpdate.AudioTranscript);
                        }

                        if (!string.IsNullOrEmpty(deltaUpdate.Text))
                        {
                            Console.Write(deltaUpdate.Text);
                            textResponse += deltaUpdate.Text;
                        }

                        // Handle function arguments
                        if (!string.IsNullOrEmpty(deltaUpdate.FunctionArguments))
                        {
                            Console.Write(deltaUpdate.FunctionArguments);

                            if (!functionArgumentBuildersById.TryGetValue(deltaUpdate.ItemId, out StringBuilder? arguments))
                            {
                                functionArgumentBuildersById[deltaUpdate.ItemId] = arguments = new();
                            }

                            arguments.Append(deltaUpdate.FunctionArguments);
                        }

                        // Handle audio
                        if (deltaUpdate.AudioBytes is not null)
                        {
                            // On first audio chunk, pause microphone to prevent echo
                            if (isFirstAudioInResponse)
                            {
                                isPlayingAudio = true;
                                isFirstAudioInResponse = false;

                                // Stop the microphone while AI is talking - only in manual mode
                                if (manualModeActive)
                                {
                                    microphoneMuted = true;
                                    if (microphoneInstance != null)
                                    {
                                        microphoneInstance.StopRecording();
                                    }
                                    Console.WriteLine("Microphone paused while AI is speaking (manual mode)");
                                }
                                else
                                {
                                    Console.WriteLine("Microphone remains active for interruptions (auto mode)");
                                }
                            }

                            // Update lastAudioOutput time
                            lastAudioOutput = DateTime.Now;

                            // Store AI audio sample for fingerprinting
                            AudioProcessor.StoreAIAudioSample(deltaUpdate.AudioBytes.ToArray());

                            // If using SpeakerOutput, play audio through it
                            speakerOutput?.EnqueueForPlayback(deltaUpdate.AudioBytes.ToArray());
                        }
                    }
                    else if (update is ConversationItemStreamingFinishedUpdate itemStreamingFinishedUpdate)
                    {
                        Console.WriteLine();
                        Console.WriteLine($"  -- Item streaming finished, item_id={itemStreamingFinishedUpdate.ItemId}");

                        // If the item was a function call, invoke the function
                        if (itemStreamingFinishedUpdate.FunctionCallId is not null)
                        {
                            Console.WriteLine($"    + Responding to tool invoked by item: {itemStreamingFinishedUpdate.FunctionName}");

                            // Parse function name
                            var (functionName, pluginName) = Helpers.ParseFunctionName(itemStreamingFinishedUpdate.FunctionName);

                            // Deserialize arguments
                            var argumentsString = functionArgumentBuildersById[itemStreamingFinishedUpdate.ItemId].ToString();
                            var arguments = Helpers.DeserializeArguments(argumentsString);

                            // Create function call content
                            var functionCallContent = new FunctionCallContent(
                                functionName: functionName,
                                pluginName: pluginName,
                                id: itemStreamingFinishedUpdate.FunctionCallId,
                                arguments: arguments);

                            try
                            {
                                // Invoke the function
                                var resultContent = await functionCallContent.InvokeAsync(kernel);

                                // Create function call output and send it back
                                ConversationItem functionOutputItem = ConversationItem.CreateFunctionCallOutput(
                                    callId: itemStreamingFinishedUpdate.FunctionCallId,
                                    output: Helpers.ProcessFunctionResult(resultContent.Result));

                                await session.AddItemAsync(functionOutputItem);
                            }
                            catch (ObjectDisposedException)
                            {
                                Console.WriteLine("Session connection closed while processing tool response.");
                                onSessionClosed?.Invoke();
                                return;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error invoking function: {ex.Message}");
                            }
                        }
                        // If the item was a message, display it
                        else if (itemStreamingFinishedUpdate.MessageContentParts?.Count > 0)
                        {
                            Console.WriteLine($"    + [{itemStreamingFinishedUpdate.MessageRole}]: ");

                            foreach (ConversationContentPart contentPart in itemStreamingFinishedUpdate.MessageContentParts)
                            {
                                Console.Write(contentPart.AudioTranscript);
                            }

                            Console.WriteLine();
                        }
                    }
                    else if (update is ConversationInputTranscriptionFinishedUpdate transcriptionUpdate)
                    {
                        Console.WriteLine();
                        Console.WriteLine($"  -- User audio transcript: {transcriptionUpdate.Transcript}");
                        Console.WriteLine();
                    }
                    else if (update is ConversationResponseFinishedUpdate turnFinishedUpdate)
                    {
                        Console.WriteLine($"  -- Model turn generation finished. Status: {turnFinishedUpdate.Status}");

                        try
                        {
                            isPlayingAudio = false;
                            lastAudioOutput = DateTime.Now;

                            // Wait for a short period to avoid feedback loop
                            await Task.Delay(audioBufferTime, ct);

                            // Check if function call results need to be processed
                            if (turnFinishedUpdate.CreatedItems.Any(item => item.FunctionName?.Length > 0))
                            {
                                Console.WriteLine("  -- Ending client turn for pending tool responses");
                                await session.StartResponseAsync();
                            }
                            else
                            {
                                // Resume microphone if it was muted
                                if (manualModeActive && microphoneMuted)
                                {
                                    microphoneMuted = false;
                                    microphoneInstance?.StartRecording();
                                    Console.WriteLine("Microphone resumed");
                                }

                                if (!string.IsNullOrEmpty(textResponse))
                                {
                                    Console.WriteLine("\n=== Complete Text Response ===");
                                    Console.WriteLine(textResponse);
                                    Console.WriteLine("==============================");
                                }

                                Console.WriteLine("\n--- AI RESPONSE COMPLETED ---");

                                // For auto mode, print a prompt to let the user know they can speak
                                if (!manualModeActive)
                                {
                                    Console.WriteLine(">>> You can speak now...");
                                }
                            }
                        }
                        catch (ObjectDisposedException)
                        {
                            Console.WriteLine("Session connection closed while processing response.");
                            onSessionClosed?.Invoke();
                            return;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error after response: {ex.Message}");
                        }
                    }
                    else if (update is ConversationErrorUpdate errorUpdate)
                    {
                        Console.WriteLine();
                        Console.WriteLine($"ERROR: {errorUpdate.Message}");

                        // Resume microphone if needed
                        if (microphoneMuted && microphoneInstance != null)
                        {
                            microphoneMuted = false;
                            microphoneInstance.StartRecording();
                            Console.WriteLine("Microphone resumed after error");
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
                Console.WriteLine("Operation was canceled.");
            }
            catch (ObjectDisposedException ex)
            {
                Console.WriteLine($"Session was disposed: {ex.Message}");
                onSessionClosed?.Invoke();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in processing updates: {ex.Message}");
            }
        }
    }
}