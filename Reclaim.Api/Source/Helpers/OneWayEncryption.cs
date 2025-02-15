using System.Security.Cryptography;
using System.Text;

namespace Reclaim.Api;

public class HashedString
{
    public string Hash;
    public string Salt;
}

public class OneWayEncryption
{
    private const int _saltBytes = 4;

    public static HashedString CreateHash(string clearText)
    {
        string salt = CreateSalt();

        return CreateHash(clearText, salt);
    }

    public static HashedString CreateHash(string clearText, string salt)
    {
        string hash = GetHash(clearText, salt);

        return new HashedString { Salt = salt, Hash = hash };
    }

    public static bool Validate(HashedString hashed, string compareTo)
    {
        string hash = GetHash(compareTo, hashed.Salt);
        return string.Compare(hashed.Hash, hash, false) == 0;
    }

    private static string GetHash(string clearText, string salt)
    {
        var hasher = SHA1.Create();
        byte[] clearTextWithSaltBytes = Encoding.UTF8.GetBytes(string.Concat(clearText, salt, Setting.OneWayEncryptionPepper));
        byte[] hashedBytes = hasher.ComputeHash(clearTextWithSaltBytes);
        hasher.Clear();

        return Convert.ToBase64String(hashedBytes);
    }

    private static string CreateSalt()
    {
        using (var rng = RandomNumberGenerator.Create())
        {
            var buff = new byte[_saltBytes];
            rng.GetBytes(buff);
            return Convert.ToBase64String(buff);
        }
    }

}