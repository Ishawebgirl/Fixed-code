using Microsoft.EntityFrameworkCore;
using System.Configuration;
using System.Runtime.Caching;

namespace Reclaim.Api;

public static class SettingManager
{
    internal static T Get<T>(SettingName settingName)
    {
        var settings = GetCachedSettings();
        var key = settingName.ToString();

        if (!settings.ContainsKey(key))
            throw new ConfigurationErrorsException($"Setting {settingName} is not defined in the settings table.");

        var value = settings[key];

        if (typeof(T).IsEnum)
            return (T)Enum.Parse(typeof(T), value);
        else
            return (T)Convert.ChangeType(value, typeof(T));
    }

    private static Dictionary<string, string> GetCachedSettings()
    {
        const string key = "Settings";
        var settings = MemoryCache.Default.Get(key) as Dictionary<string, string>;

        if (settings == null)
        {
            settings = new Dictionary<string, string>();

            try
            {
                using (var db = new Model.DatabaseContext(DbContextOptions.Get()))
                {
                    foreach (var setting in db.ApplicationSettings.AsNoTracking())
                    {
                        var name = setting.Name;
                        var value = setting.Value;
                        var isSecret = setting.IsSecret;

                        if (isSecret == true)
                            value = TwoWayEncryption.Decrypt(value);

                        settings.Add(name, value);
                    }

                    db.SaveChanges();
                }

                MemoryCache.Default.Set(key, settings, DateTime.UtcNow.AddSeconds(Constant.SettingCacheTimeout));
            }
            catch (Exception ex)
            {
                throw new ApiException(Model.ErrorCode.ApplicationSettingsInvalid, $"Failed to load application settings from database.  {ex.Message}");
            }
        }

        return settings;
    }
}