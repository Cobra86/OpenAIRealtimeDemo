using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using NAudio.Wave;
using OpenAI.RealtimeConversation;
using System.ClientModel;
using System.ComponentModel;
using System.Text;
using System.Text.Json;

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
        static readonly TimeSpan audioQuietPeriod = TimeSpan.FromSeconds(2.0); // Extended quiet period
        static readonly TimeSpan audioBufferTime = TimeSpan.FromMilliseconds(500); // Buffer time after audio

        public static async Task Main(string[] args)
        {
            var config = new ConfigurationBuilder()
                .AddUserSecrets<Program>()
                .Build();

            string? apiKey = config["OpenAIKey"]; 

            if (string.IsNullOrWhiteSpace(apiKey))
            {
                Console.WriteLine("API key not found. Please set OpenAI:ApiKey in configuration or environment variable.");
                return;
            }

            // Build kernel with plugins
            var kernel = Kernel.CreateBuilder().Build();
            kernel.ImportPluginFromType<WeatherPlugin>();

            // Initialize Realtime Conversation client
            var realtimeClient = GetRealtimeConversationClient(apiKey);

            /* Select microphone device */
            int deviceCount = WaveInEvent.DeviceCount;
            Console.WriteLine("Available microphones:");
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
                    foreach (var tool in ConvertFunctions(kernel))
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

                    // Start the message receiver task
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
                    // Ensure the session is always properly disposed
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

                    session.Dispose();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }

        // Helper to process session updates asynchronously
        private static async Task ProcessSessionUpdatesAsync(RealtimeConversationSession session, Kernel kernel,
                                                          CancellationToken ct, Action onSessionClosed = null)
        {
            // For incremental audio playback
            using var outDev = new WaveOutEvent();
            var bufProv = new BufferedWaveProvider(new WaveFormat(SR, BPS, CH))
            { DiscardOnBufferOverflow = true };
            outDev.Init(bufProv);
            outDev.Play();
            Console.WriteLine("Audio output initialized");

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
                            Console.Write($"    {itemStreamingStartedUpdate.FunctionName}: ");
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
                                if (microphoneInstance != null && manualModeActive)
                                {
                                    microphoneMuted = true;
                                    try
                                    {
                                        microphoneInstance.StopRecording();
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Error stopping microphone: {ex.Message}");
                                    }
                                    Console.WriteLine("Microphone paused while AI is speaking (manual mode)");
                                }

                                // In auto mode, we keep mic active but store AI audio fingerprint
                                if (!manualModeActive)
                                {
                                    Console.WriteLine("Microphone remains active for interruptions (auto mode)");
                                }
                            }

                            // Update lastAudioOutput time
                            lastAudioOutput = DateTime.Now;

                            // Store AI audio sample for fingerprinting
                            AudioProcessor.StoreAIAudioSample(deltaUpdate.AudioBytes.ToArray());

                            if (collectingAudioResponse)
                            {
                                // Add to buffer for later playback
                                audioResponseBuffer.Add(deltaUpdate.AudioBytes.ToArray());
                                Console.Write(".");  // Show progress
                            }
                            else
                            {
                                // Play incrementally
                                bufProv.AddSamples(deltaUpdate.AudioBytes.ToArray(), 0, deltaUpdate.AudioBytes.ToArray().Length);
                            }
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
                            var (functionName, pluginName) = ParseFunctionName(itemStreamingFinishedUpdate.FunctionName);

                            // Deserialize arguments
                            var argumentsString = functionArgumentBuildersById[itemStreamingFinishedUpdate.ItemId].ToString();
                            var arguments = DeserializeArguments(argumentsString);

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
                                    output: ProcessFunctionResult(resultContent.Result));

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
                            Console.Write($"    + [{itemStreamingFinishedUpdate.MessageRole}]: ");

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
                            // Play collected audio if needed
                            if (collectingAudioResponse)
                            {
                                collectingAudioResponse = false;
                                isPlayingAudio = true;
                                await PlayCollectedAudio();
                                // Note: PlayCollectedAudio will set isPlayingAudio = false when done
                                // and will restart microphone in manual mode
                            }
                            else
                            {
                                isPlayingAudio = false;
                                lastAudioOutput = DateTime.Now;

                                // For auto mode, wait for a short period to avoid feedback loop
                                if (!manualModeActive)
                                {
                                    await Task.Delay(audioBufferTime);
                                }
                            }

                            // Check if function call results need to be processed
                            if (turnFinishedUpdate.CreatedItems.Any(item => item.FunctionName?.Length > 0))
                            {
                                Console.WriteLine("  -- Ending client turn for pending tool responses");
                                await session.StartResponseAsync();
                            }
                            else
                            {
                                // Only resume microphone here for auto mode - manual mode is handled in PlayCollectedAudio
                                // or earlier in this block for non-collected audio
                                if (microphoneInstance != null && (!manualModeActive || !collectingAudioResponse))
                                {
                                    microphoneMuted = false;
                                    try
                                    {
                                        microphoneInstance.StartRecording();
                                        Console.WriteLine("Microphone activated - ready for next input");
                                    }
                                    catch (InvalidOperationException ex)
                                    {
                                        Console.WriteLine($"Microphone already recording: {ex.Message}");
                                    }
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
                        if (microphoneInstance != null)
                        {
                            microphoneMuted = false;
                            try
                            {
                                microphoneInstance.StartRecording();
                            }
                            catch (InvalidOperationException ex)
                            {
                                Console.WriteLine($"Microphone already recording: {ex.Message}");
                            }
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
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

        // Helper to play collected audio
        private static Task PlayCollectedAudio()
        {
            return Task.Run(async () =>
            {
                try
                {
                    if (audioResponseBuffer.Count == 0)
                    {
                        Console.WriteLine("No audio to play.");
                        isPlayingAudio = false;
                        return;
                    }

                    Console.WriteLine($"Playing complete audio response ({audioResponseBuffer.Count} chunks)...");

                    // Calculate total size
                    int totalSize = 0;
                    foreach (var chunk in audioResponseBuffer)
                    {
                        totalSize += chunk.Length;
                    }

                    // Combine chunks
                    byte[] completeAudio = new byte[totalSize];
                    int position = 0;

                    foreach (var chunk in audioResponseBuffer)
                    {
                        Array.Copy(chunk, 0, completeAudio, position, chunk.Length);
                        position += chunk.Length;
                    }

                    // Play audio
                    using (var ms = new MemoryStream(completeAudio))
                    using (var reader = new RawSourceWaveStream(ms, new WaveFormat(SR, BPS, CH)))
                    using (var waveOut = new WaveOutEvent())
                    {
                        waveOut.Init(reader);
                        waveOut.Play();

                        // Wait for playback to complete
                        while (waveOut.PlaybackState == PlaybackState.Playing)
                        {
                            Thread.Sleep(100);
                        }
                    }

                    Console.WriteLine("Audio playback completed.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error playing audio: {ex.Message}");
                }
                finally
                {
                    // Clear buffer after playing
                    audioResponseBuffer.Clear();
                    isPlayingAudio = false;

                    // Record the time when audio output finished
                    lastAudioOutput = DateTime.Now;

                    // Add a delay after audio finishes to prevent feedback loop
                    await Task.Delay(audioBufferTime);  // Add buffer time after audio playback

                    // Only resume microphone in manual mode - auto mode is handled in ResponseFinishedUpdate
                    if (microphoneInstance != null && manualModeActive)
                    {
                        microphoneMuted = false;
                        try
                        {
                            microphoneInstance.StartRecording();
                            Console.WriteLine("Microphone resumed after playback");
                        }
                        catch (InvalidOperationException ex)
                        {
                            Console.WriteLine($"Microphone already recording: {ex.Message}");
                        }
                    }
                }
            });
        }

        /// <summary>A sample plugin to get weather information.</summary>
        private sealed class WeatherPlugin
        {
            [KernelFunction]
            [Description("Gets the current weather for the specified city in Fahrenheit.")]
            public static string GetWeatherForCity([Description("City name without state/country.")] string cityName)
            {
                return cityName switch
                {
                    "Boston" => "61 and rainy",
                    "London" => "55 and cloudy",
                    "Miami" => "80 and sunny",
                    "Paris" => "60 and rainy",
                    "Tokyo" => "50 and sunny",
                    "Sydney" => "75 and sunny",
                    "Tel Aviv" => "80 and sunny",
                    "San Francisco" => "70 and sunny",
                    _ => throw new ArgumentException($"Data is not available for {cityName}."),
                };
            }
        }

        #region Helper Methods

        /// <summary>Helper method to get a RealtimeConversationClient.</summary>
        private static RealtimeConversationClient GetRealtimeConversationClient(string apiKey)
        {
            return new RealtimeConversationClient(
                model: "gpt-4o-realtime-preview",
                credential: new ApiKeyCredential(apiKey));
        }

        /// <summary>Helper method to parse a function name for compatibility with Semantic Kernel plugins/functions.</summary>
        private static (string FunctionName, string? PluginName) ParseFunctionName(string fullyQualifiedName)
        {
            const string FunctionNameSeparator = "-";

            string? pluginName = null;
            string functionName = fullyQualifiedName;

            int separatorPos = fullyQualifiedName.IndexOf(FunctionNameSeparator, StringComparison.Ordinal);
            if (separatorPos >= 0)
            {
                pluginName = fullyQualifiedName.AsSpan(0, separatorPos).Trim().ToString();
                functionName = fullyQualifiedName.AsSpan(separatorPos + FunctionNameSeparator.Length).Trim().ToString();
            }

            return (functionName, pluginName);
        }

        /// <summary>Helper method to deserialize function arguments.</summary>
        private static KernelArguments? DeserializeArguments(string argumentsString)
        {
            var arguments = JsonSerializer.Deserialize<KernelArguments>(argumentsString);

            if (arguments is not null)
            {
                // Iterate over copy of the names to avoid mutating the dictionary while enumerating it
                var names = arguments.Names.ToArray();
                foreach (var name in names)
                {
                    arguments[name] = arguments[name]?.ToString();
                }
            }

            return arguments;
        }

        /// <summary>Helper method to process function result in order to provide it to the model as string.</summary>
        private static string? ProcessFunctionResult(object? functionResult)
        {
            if (functionResult is string stringResult)
            {
                return stringResult;
            }

            return JsonSerializer.Serialize(functionResult);
        }

        /// <summary>Helper method to convert Kernel plugins/function to realtime session conversation tools.</summary>
        private static IEnumerable<ConversationTool> ConvertFunctions(Kernel kernel)
        {
            foreach (var plugin in kernel.Plugins)
            {
                var functionsMetadata = plugin.GetFunctionsMetadata();

                foreach (var metadata in functionsMetadata)
                {
                    var toolDefinition = metadata.ToOpenAIFunction().ToFunctionDefinition(false);

                    yield return new ConversationFunctionTool(name: toolDefinition.FunctionName)
                    {
                        Description = toolDefinition.FunctionDescription,
                        Parameters = toolDefinition.FunctionParameters
                    };
                }
            }
        }

        #endregion
    }
}