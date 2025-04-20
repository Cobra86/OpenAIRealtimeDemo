using Microsoft.SemanticKernel;
using System.ComponentModel;

namespace OpenAIRealtimeDemo
{
    /// <summary>A sample plugin to get weather information.</summary>
    internal sealed class WeatherPlugin
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
}
