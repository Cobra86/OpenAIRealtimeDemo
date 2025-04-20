using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using NAudio.Wave;
using OpenAI.RealtimeConversation;

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
            var realtimeClient = Helpers.GetRealtimeConversationClient(apiKey);

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
                    _ = Task.Run(() => new AudioProcessor().ProcessSessionUpdatesAsync(session, kernel, cts.Token, () => sessionActive = false));

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
    }
}