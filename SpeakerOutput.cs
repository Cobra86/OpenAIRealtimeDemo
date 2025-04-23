using NAudio.Wave;
using System;
namespace OpenAIRealtimeDemo
{
    /// <summary>
    /// Uses the NAudio library (https://github.com/naudio/NAudio) to provide a rudimentary abstraction to output
    /// audio segments to the default output (speaker/headphone) device.
    /// </summary>
    public class SpeakerOutput : IDisposable
    {
        private BufferedWaveProvider _waveProvider;
        private WaveOutEvent _waveOutEvent;
        public SpeakerOutput()
        {
            WaveFormat outputAudioFormat = new(
                rate: 24000,
                bits: 16,
                channels: 1);
            _waveProvider = new BufferedWaveProvider(outputAudioFormat)
            {
                BufferDuration = TimeSpan.FromMinutes(2),
            };
            _waveOutEvent = new WaveOutEvent();
            _waveOutEvent.Init(_waveProvider);  // Fixed line
            _waveOutEvent.Play();
        }
        public void EnqueueForPlayback(byte[] audioData)
        {
            if (audioData != null && audioData.Length > 0)
            {
                _waveProvider.AddSamples(audioData, 0, audioData.Length);
            }
        }
        public void ClearPlayback()
        {
            _waveProvider.ClearBuffer();
        }
        public void Dispose()
        {
            _waveOutEvent?.Dispose();
        }
    }
}