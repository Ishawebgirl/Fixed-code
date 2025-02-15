using System.Security.Cryptography;

namespace Reclaim.Api;

public class TwoWayEncryption
{
    private const int saltLength = 4;
    private static byte[] password = new byte[] { 225, 91, 124, 120, 134, 236, 45, 221, 91, 76, 36, 222, 183, 15, 198, 106, 91, 185, 221, 235, 203, 240, 226, 183, 214, 244, 153, 11, 115, 80, 15, 185, 66, 147, 160, 87, 244, 134, 103, 226, 46, 168, 177, 95, 73, 178, 32, 24, 186, 101, 161, 199, 225, 243, 37, 103, 30, 88, 180, 88, 66, 4, 177, 18, 98 };
    private static byte[] passwordSalt = new byte[] { 184, 227, 36, 77, 199, 238, 134, 56 };

    private static string CreateSalt()
    {
        var r = CreateRandomBytes(saltLength);

        return Convert.ToBase64String(r);
    }

    private static byte[] CreateRandomBytes(int len)
    {
        var r = new byte[len];
        RandomNumberGenerator.Create().GetBytes(r);

        return r;
    }

    public static string Encrypt(string clearText)
    {
        var salt = CreateSalt();
        var clearBytes = System.Text.Encoding.Unicode.GetBytes(salt + clearText);
        var pdb = new PasswordDeriveBytes(password, passwordSalt);
        var encryptedData = Encrypt(clearBytes, pdb.GetBytes(32), pdb.GetBytes(16));

        return Convert.ToBase64String(encryptedData);
    }

    public static byte[] Encrypt(byte[] clearData, byte[] Key, byte[] IV)
    {
        var ms = new MemoryStream();
        var alg = Aes.Create();

        alg.Key = Key;
        alg.IV = IV;
        var cs = new CryptoStream(ms, alg.CreateEncryptor(), CryptoStreamMode.Write);

        cs.Write(clearData, 0, clearData.Length);
        cs.Close();
        var encryptedData = ms.ToArray();
        return encryptedData;
    }

    public static string Decrypt(string cipherText)
    {
        var cipherBytes = Convert.FromBase64String(cipherText);
        var pdb = new PasswordDeriveBytes(password, passwordSalt);
        var decryptedData = Decrypt(cipherBytes, pdb.GetBytes(32), pdb.GetBytes(16));
        var saltedText = System.Text.Encoding.Unicode.GetString(decryptedData);

        return saltedText.Substring(saltLength * 2, saltedText.Length - saltLength * 2);
    }

    public static byte[] Decrypt(byte[] cipherData, byte[] Key, byte[] IV)
    {
        var ms = new MemoryStream();
        var alg = Aes.Create();
        alg.Key = Key;
        alg.IV = IV;
        var cs = new CryptoStream(ms, alg.CreateDecryptor(), CryptoStreamMode.Write);
        cs.Write(cipherData, 0, cipherData.Length);
        cs.Close();
        var decryptedData = ms.ToArray();

        return decryptedData;
    }
}
