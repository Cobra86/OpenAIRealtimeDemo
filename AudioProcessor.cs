using System;
using System.Collections.Generic;

namespace OpenAIRealtimeDemo
{
    /// <summary>
    /// Handles audio processing and voice detection to prevent feedback loops
    /// </summary>
    public static class AudioProcessor
    {
        // AI voice fingerprinting settings
        private static readonly double amplitudeThreshold = 0.4; // Threshold to detect significant audio
        private static readonly int voiceFrequencyMin = 85; // Typical AI voice frequency range (Hz)
        private static readonly int voiceFrequencyMax = 255; // Upper end of AI voice
        private static readonly Queue<short> recentAudioSamples = new Queue<short>(1000); // Store recent AI audio for comparison

        /// <summary>
        /// Determines if provided audio data is likely from the AI voice output
        /// </summary>
        public static bool IsLikelyAIVoice(byte[] audioData)
        {
            try
            {
                // This is a simplified approach that checks amplitude patterns
                // A real solution would use more sophisticated audio fingerprinting

                if (audioData.Length < 100) return false;

                // Convert byte array to shorts (since we're using 16-bit PCM)
                short[] samples = new short[audioData.Length / 2];
                for (int i = 0; i < samples.Length; i++)
                {
                    if (i * 2 + 1 < audioData.Length)
                    {
                        samples[i] = BitConverter.ToInt16(audioData, i * 2);
                    }
                }

                // Calculate average amplitude
                double sum = 0;
                foreach (short sample in samples)
                {
                    sum += Math.Abs(sample);
                }
                double avgAmplitude = sum / samples.Length;

                // Simple amplitude-based detection - not perfect but better than nothing
                double normalizedAmplitude = avgAmplitude / short.MaxValue;

                // If we have enough reference samples, compare patterns
                if (recentAudioSamples.Count >= 500)
                {
                    // Compare to recent AI audio pattern
                    // This is very simplified - real solutions would use spectral analysis
                    short[] reference = recentAudioSamples.ToArray();
                    double similarity = 0;

                    // Basic pattern similarity check
                    for (int i = 0; i < Math.Min(100, samples.Length); i++)
                    {
                        similarity += Math.Abs(samples[i] - reference[i % reference.Length]);
                    }

                    similarity /= 100 * short.MaxValue;

                    // If very similar to recent AI audio, it's likely feedback
                    if (similarity < 0.3)
                    {
                        return true;
                    }
                }

                return normalizedAmplitude < amplitudeThreshold;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in voice detection: {ex.Message}");
                return false; // On error, assume it's not AI voice
            }
        }

        /// <summary>
        /// Stores audio samples from AI for fingerprinting and comparison
        /// </summary>
        public static void StoreAIAudioSample(byte[] audioData)
        {
            try
            {
                // Store samples for comparison later
                for (int i = 0; i < audioData.Length; i += 2)
                {
                    if (i + 1 < audioData.Length)
                    {
                        short sample = BitConverter.ToInt16(audioData, i);

                        // Keep queue at maximum size
                        if (recentAudioSamples.Count >= 1000)
                        {
                            recentAudioSamples.Dequeue();
                        }

                        recentAudioSamples.Enqueue(sample);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error storing audio sample: {ex.Message}");
                // Continue execution even if we can't store the sample
            }
        }
    }
}