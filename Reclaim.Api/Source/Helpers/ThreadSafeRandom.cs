namespace Reclaim.Api;

public static class ThreadSafeRandom
{
    private static Random _random = new Random();

    public static int Next(int minValue, int maxValue)
    {
        lock (_random)
        {
            return _random.Next(minValue, maxValue);
        }
    }

    public static string GetRandomCharacters(string field, int len)
    {
        var chars = new char[len];

        for (int i = 0; i < len; i++)
            chars[i] = field[Next(0, field.Length - 1)];

        return new string(chars);
    }
}