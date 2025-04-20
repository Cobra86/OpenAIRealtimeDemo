using Microsoft.SemanticKernel;
using NAudio.Wave;
using OpenAI.RealtimeConversation;
using System.Text;

namespace OpenAIRealtimeDemo
{
#pragma warning disable OPENAI002 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    public class AudioProcessor
    {
        // Audio constants
        private const int SR = 24_000;   // sample‑rate Hz
        private const int BPS = 16;      // bits per sample
        private const int CH = 1;        // mono PCM

        // Session state tracking
        private bool isPlayingAudio = false;
        private bool microphoneMuted = false;
        private WaveInEvent? microphoneInstance = null;

        // Audio response handling
        private List<byte[]> audioResponseBuffer = new List<byte[]>();
        private bool collectingAudioResponse = false;
        private DateTime lastAudioOutput = DateTime.MinValue;
        private readonly TimeSpan audioQuietPeriod = TimeSpan.FromSeconds(2.0);
        private readonly TimeSpan audioBufferTime = TimeSpan.FromMilliseconds(500);

        // Operation mode
        private bool manualModeActive = true;

        // Voice detection
        private static readonly double amplitudeThreshold = 0.4;
        private static readonly int voiceFrequencyMin = 85;
        private static readonly int voiceFrequencyMax = 255;
        private static readonly Queue<short> recentAudioSamples = new Queue<short>(1000);
        private static DateTime lastSampleTime = DateTime.MinValue;

        // Events
        public event EventHandler<string> LogMessage;
        public event EventHandler MicrophoneResumed;
        public event EventHandler MicrophoneStopped;

        public AudioProcessor(bool manualMode = true)
        {
            manualModeActive = manualMode;
        }

        public bool IsPlaying => isPlayingAudio;
        public bool IsMicrophoneMuted => microphoneMuted;
        public bool IsCollectingAudio => collectingAudioResponse;

        public void SetMicrophone(WaveInEvent microphone)
        {
            microphoneInstance = microphone;
        }

        public void StartCollectingAudioResponse()
        {
            audioResponseBuffer.Clear();
            collectingAudioResponse = true;
        }

        public static bool IsLikelyAIVoice(byte[] audioData, bool isAIOutput = false)
        {
            if (isAIOutput) return false;
            try
            {
                if (audioData.Length < 100) return false;
                short[] samples = ConvertToShortArray(audioData);
                double normalizedAmplitude = CalculateNormalizedAmplitude(samples);
                if (recentAudioSamples.Count >= 500)
                {
                    double similarity = CalculateSimilarity(samples, recentAudioSamples.ToArray());

                    Console.WriteLine($"[AudioProcessor] Similarity to AI audio: {similarity:F2}");
                    if (similarity < 0.1)
                    {
                        Console.WriteLine("[AudioProcessor] Audio flagged as likely AI feedback, skipping.");
                        return true;
                    }
                }
                return normalizedAmplitude < amplitudeThreshold;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in voice detection: {ex.Message}");
                return false;
            }
        }

        public static void StoreAIAudioSample(byte[] audioData)
        {
            if ((DateTime.Now - lastSampleTime).TotalMilliseconds < 100) return;
            lastSampleTime = DateTime.Now;
            try
            {
                foreach (short sample in ConvertToShortArray(audioData))
                {
                    if (recentAudioSamples.Count >= 1000)
                    {
                        recentAudioSamples.Dequeue();
                    }
                    recentAudioSamples.Enqueue(sample);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error storing audio sample: {ex.Message}");
            }
        }

        private static short[] ConvertToShortArray(byte[] audioData)
        {
            short[] samples = new short[audioData.Length / 2];
            for (int i = 0; i < samples.Length; i++)
            {
                if (i * 2 + 1 < audioData.Length)
                {
                    samples[i] = BitConverter.ToInt16(audioData, i * 2);
                }
            }
            return samples;
        }

        private static double CalculateNormalizedAmplitude(short[] samples)
        {
            double sum = 0;
            foreach (short sample in samples)
            {
                sum += Math.Abs(sample);
            }
            return sum / samples.Length / short.MaxValue;
        }

        private static double CalculateSimilarity(short[] samples, short[] reference)
        {
            double similarity = 0;
            for (int i = 0; i < Math.Min(100, samples.Length); i++)
            {
                similarity += Math.Abs(samples[i] - reference[i % reference.Length]) / (i + 1);
            }
            return similarity / 100 * short.MaxValue;
        }

        public void ProcessAudioChunk(byte[] audioChunk)
        {
            if (collectingAudioResponse)
            {
                // Add to buffer for later playback
                audioResponseBuffer.Add(audioChunk);
                LogInfo(".");  // Show progress
            }
            else
            {
                // Store for voice detection
                StoreAIAudioSample(audioChunk);

                // Update timestamp
                lastAudioOutput = DateTime.Now;
            }
        }

        public void PauseMicrophone()
        {
            if (microphoneInstance != null)
            {
                microphoneMuted = true;
                try
                {
                    microphoneInstance.StopRecording();
                    LogInfo("Microphone paused");
                    MicrophoneStopped?.Invoke(this, EventArgs.Empty);
                }
                catch (Exception ex)
                {
                    LogError($"Error stopping microphone: {ex.Message}");
                }
            }
        }

        public void ResumeMicrophone()
        {
            if (microphoneInstance != null)
            {
                microphoneMuted = false;
                try
                {
                    microphoneInstance.StartRecording();
                    LogInfo("Microphone resumed");
                    MicrophoneResumed?.Invoke(this, EventArgs.Empty);
                }
                catch (InvalidOperationException ex)
                {
                    LogError($"Microphone already recording: {ex.Message}");
                }
            }
        }

        // Helper to play collected audio
        public async Task PlayCollectedAudio()
        {
            try
            {
                if (audioResponseBuffer.Count == 0)
                {
                    LogInfo("No audio to play.");
                    isPlayingAudio = false;
                    return;
                }

                LogInfo($"Playing complete audio response ({audioResponseBuffer.Count} chunks)...");

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
                        await Task.Delay(100);
                    }
                }

                LogInfo("Audio playback completed.");
            }
            catch (Exception ex)
            {
                LogError($"Error playing audio: {ex.Message}");
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
                if (manualModeActive)
                {
                    ResumeMicrophone();
                }
            }
        }

        // Helper to process session updates asynchronously
        public async Task ProcessSessionUpdatesAsync(RealtimeConversationSession session, Kernel kernel,
                                                  CancellationToken ct, Action onSessionClosed = null)
        {
            // For incremental audio playback
            using var outDev = new WaveOutEvent();
            var bufProv = new BufferedWaveProvider(new WaveFormat(SR, BPS, CH))
            { DiscardOnBufferOverflow = true };
            outDev.Init(bufProv);
            outDev.Play();
            LogInfo("Audio output initialized");

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
                        LogInfo($"<<< Session started. ID: {sessionStartedUpdate.SessionId}");
                    }
                    else if (update is ConversationInputSpeechStartedUpdate)
                    {
                        LogInfo("  -- Voice activity detection started");
                    }
                    else if (update is ConversationInputSpeechFinishedUpdate)
                    {
                        LogInfo("  -- Voice activity detection ended");
                    }
                    else if (update is ConversationItemStreamingStartedUpdate itemStreamingStartedUpdate)
                    {
                        isFirstAudioInResponse = true;
                        textResponse = "";

                        LogInfo("  -- Begin streaming of new item");
                        if (!string.IsNullOrEmpty(itemStreamingStartedUpdate.FunctionName))
                        {
                            LogInfo($"    {itemStreamingStartedUpdate.FunctionName}: ");
                        }
                    }
                    else if (update is ConversationItemStreamingPartDeltaUpdate deltaUpdate)
                    {
                        // Handle transcription or text
                        if (!string.IsNullOrEmpty(deltaUpdate.AudioTranscript))
                        {
                            LogInfo(deltaUpdate.AudioTranscript, appendNewLine: false);
                        }

                        if (!string.IsNullOrEmpty(deltaUpdate.Text))
                        {
                            LogInfo(deltaUpdate.Text, appendNewLine: false);
                            textResponse += deltaUpdate.Text;
                        }

                        // Handle function arguments
                        if (!string.IsNullOrEmpty(deltaUpdate.FunctionArguments))
                        {
                            LogInfo(deltaUpdate.FunctionArguments, appendNewLine: false);

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
                                    PauseMicrophone();
                                    LogInfo("Microphone paused while AI is speaking (manual mode)");
                                }
                                else
                                {
                                    LogInfo("Microphone remains active for interruptions (auto mode)");
                                }
                            }

                            // Update lastAudioOutput time
                            lastAudioOutput = DateTime.Now;

                            // Store AI audio sample for fingerprinting
                            StoreAIAudioSample(deltaUpdate.AudioBytes.ToArray());

                            // Process the audio chunk
                            ProcessAudioChunk(deltaUpdate.AudioBytes.ToArray());

                            // If not collecting for buffer, play incrementally
                            if (!collectingAudioResponse)
                            {
                                bufProv.AddSamples(deltaUpdate.AudioBytes.ToArray(), 0, deltaUpdate.AudioBytes.ToArray().Length);
                            }
                        }
                    }
                    else if (update is ConversationItemStreamingFinishedUpdate itemStreamingFinishedUpdate)
                    {
                        LogInfo("", true);
                        LogInfo($"  -- Item streaming finished, item_id={itemStreamingFinishedUpdate.ItemId}");

                        // If the item was a function call, invoke the function
                        if (itemStreamingFinishedUpdate.FunctionCallId is not null)
                        {
                            LogInfo($"    + Responding to tool invoked by item: {itemStreamingFinishedUpdate.FunctionName}");

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
                                LogInfo("Session connection closed while processing tool response.");
                                onSessionClosed?.Invoke();
                                return;
                            }
                            catch (Exception ex)
                            {
                                LogError($"Error invoking function: {ex.Message}");
                            }
                        }
                        // If the item was a message, display it
                        else if (itemStreamingFinishedUpdate.MessageContentParts?.Count > 0)
                        {
                            LogInfo($"    + [{itemStreamingFinishedUpdate.MessageRole}]: ", appendNewLine: false);

                            foreach (ConversationContentPart contentPart in itemStreamingFinishedUpdate.MessageContentParts)
                            {
                                LogInfo(contentPart.AudioTranscript, appendNewLine: false);
                            }

                            LogInfo("", true);
                        }
                    }
                    else if (update is ConversationInputTranscriptionFinishedUpdate transcriptionUpdate)
                    {
                        LogInfo("", true);
                        LogInfo($"  -- User audio transcript: {transcriptionUpdate.Transcript}");
                        LogInfo("", true);
                    }
                    else if (update is ConversationResponseFinishedUpdate turnFinishedUpdate)
                    {
                        LogInfo($"  -- Model turn generation finished. Status: {turnFinishedUpdate.Status}");

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
                                LogInfo("  -- Ending client turn for pending tool responses");
                                await session.StartResponseAsync();
                            }
                            else
                            {
                                // Only resume microphone here for auto mode - manual mode is handled in PlayCollectedAudio
                                // or earlier in this block for non-collected audio
                                if (!manualModeActive || !collectingAudioResponse)
                                {
                                    ResumeMicrophone();
                                }

                                if (!string.IsNullOrEmpty(textResponse))
                                {
                                    LogInfo("\n=== Complete Text Response ===");
                                    LogInfo(textResponse);
                                    LogInfo("==============================");
                                }

                                LogInfo("\n--- AI RESPONSE COMPLETED ---");

                                // For auto mode, print a prompt to let the user know they can speak
                                if (!manualModeActive)
                                {
                                    LogInfo(">>> You can speak now...");
                                }
                            }
                        }
                        catch (ObjectDisposedException)
                        {
                            LogInfo("Session connection closed while processing response.");
                            onSessionClosed?.Invoke();
                            return;
                        }
                        catch (Exception ex)
                        {
                            LogError($"Error after response: {ex.Message}");
                        }
                    }
                    else if (update is ConversationErrorUpdate errorUpdate)
                    {
                        LogInfo("", true);
                        LogError($"ERROR: {errorUpdate.Message}");

                        // Resume microphone if needed
                        ResumeMicrophone();
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
                LogInfo("Operation was canceled.");
            }
            catch (ObjectDisposedException ex)
            {
                LogError($"Session was disposed: {ex.Message}");
                onSessionClosed?.Invoke();
            }
            catch (Exception ex)
            {
                LogError($"Error in processing updates: {ex.Message}");
            }
        }

        private void LogInfo(string message, bool appendNewLine = true)
        {
            if (appendNewLine)
            {
                Console.WriteLine(message);
            }
            else
            {
                Console.Write(message);
            }

            LogMessage?.Invoke(this, message);
        }

        private void LogError(string message)
        {
            Console.WriteLine(message);
            LogMessage?.Invoke(this, $"ERROR: {message}");
        }
    }
}