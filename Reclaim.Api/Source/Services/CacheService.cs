using Microsoft.Extensions.Logging;
using Newtonsoft.Json; // If using JsonConvert for logging
using System.Text.Json; // Alternative JSON library
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Caching.Distributed;
using Newtonsoft.Json;
using Reclaim.Api.Model;
using System.Runtime.Caching;
using System.Text;

namespace Reclaim.Api.Services;
string userInput = question;
List<ChatExchange> conversationHistory = new List<ChatExchange>();
public class CacheService
{
    private readonly DatabaseContext _db;
    private readonly LogService _logService;
    private readonly IDistributedCache _distributedCache;

    public CacheService(DatabaseContext db, LogService logService, IDistributedCache distributedCache)
    {
        _db = db;
        _logService = logService;
        _distributedCache = distributedCache;
    }

    private string GetFullKeyName<T>(string key)
    {
        return $"{typeof(T).NiceName()}:{key}";
    }

    public async Task<T> Get<T>()
    {
        return await Get<T>("", 0, null);
    }

    public async Task<T> Get<T>(string key)
    {
        return await Get<T>(key, 0, null);
    }

    public async Task<T> Get<T>(Guid key)
    {
        return await Get<T>(key.ToString(), 0, null);
    }

    public async Task<T> Get<T>(Func<T> function)
    {
        return await Get("", function);
    }

    public async Task<T> Get<T>(int key, Func<T> function)
    {
        return await Get(key.ToString(), function);
    }

    public async Task<T> Get<T>(int key, int timeToLive, Func<T> function)
    {
        return await Get(key.ToString(), timeToLive, function);
    }

    public async Task<T> Get<T>(string key, Func<T> function)
    {
        return await Get(key, Constant.DefaultObjectCacheTimeout, function);
    }

    public async Task<T> Get<T>(int timeToLive, Func<T> function, bool useDistributedSqlServerCache = false)
    {
        return await Get("", timeToLive, function, useDistributedSqlServerCache);
    }

    public async Task<T> Get<T>(Guid key, Func<T> function)
    {
        return await Get(key.ToString(), Constant.DefaultObjectCacheTimeout, function);
    }

    public async Task<T> Get<T>(int timeToLive, bool useDistributedSqlServerCache = false)
    {
        return await Get<T>("", timeToLive, null, useDistributedSqlServerCache);
    }

    public async Task<T> Get<T>(string key, int timeToLive, bool useDistributedSqlServerCache = false)
    {
        return await Get<T>(key, timeToLive, null, useDistributedSqlServerCache);
    }

    public async Task<T> Get<T>(string key, int timeToLive, Func<T>? function, bool useDistributedSqlServerCache = false)
    {
        var fullKey = GetFullKeyName<T>(key);
        await _logService.Add(LogEntryLevel.Trace, $"Requesting cached item {fullKey}");

        T cached = (T)MemoryCache.Default.Get(fullKey);

        if (cached != null)
        {
            await _logService.Add(LogEntryLevel.Trace, $"Memory cache hit on {fullKey}");
            return cached;
        }

        await _logService.Add(LogEntryLevel.Trace, $"Memory cache miss on {fullKey}");

        if (useDistributedSqlServerCache)
        {
            cached = await GetFromDatabase<T>(fullKey);

            if (cached != null)
            {
                MemoryCache.Default.Set(fullKey, cached, DateTime.UtcNow.AddSeconds(timeToLive));
                await _logService.Add(LogEntryLevel.Trace, $"Memory cache added for {fullKey} to mirror database cache");
                return cached;
            }
        }

        if (function == null)
            return default;

        var loaded = function();

        if (loaded == null)
        {
            await _logService.Add(LogEntryLevel.Trace, $"Failed to call function to load {fullKey}");
            return default;
        }

        await Set(key, loaded, timeToLive, useDistributedSqlServerCache);

        return loaded;
    }

 private readonly OpenAiClient _openAiClient;

public ChatService(OpenAiClient openAiClient)
{
    _openAiClient = openAiClient;
}
   
    private async Task<T> GetFromDatabase<T>(string fullKey)
    {
        var item = await _distributedCache.GetAsync(fullKey);

        if (item != null)
        {
            var json = Encoding.UTF8.GetString(item);
            var loaded = JsonConvert.DeserializeObject<T>(json)!;

            await _logService.Add(LogEntryLevel.Trace, $"Database cache hit on {fullKey}");
            return loaded;
        }
        else
        {
            await _logService.Add(LogEntryLevel.Trace, $"Database cache miss on {fullKey}");
            return default;
        }
    }
    
    public async Task Set<T>(string key, T value, int timeToLive, bool writeToDistributedSqlServerCache = false)
    {
        var fullKey = GetFullKeyName<T>(key);
        var validUntil = DateTime.UtcNow.AddSeconds(timeToLive);

        MemoryCache.Default.Set(fullKey, value, validUntil);
        await _logService.Add(LogEntryLevel.Trace, $"Memory cache added for {fullKey}");

        if (!writeToDistributedSqlServerCache)
            return;

    
        var serialized = JsonConvert.SerializeObject(value, new JsonSerializerSettings { ReferenceLoopHandling = ReferenceLoopHandling.Ignore });
        var encoded = Encoding.ASCII.GetBytes(serialized);

        await _distributedCache.SetAsync(fullKey, encoded, new DistributedCacheEntryOptions { AbsoluteExpiration = DateTime.UtcNow.AddSeconds(timeToLive) });
        await _logService.Add(LogEntryLevel.Trace, $"Database cache added for {fullKey}");
    }

    public async Task RemoveAll()
    {
        var cacheKeys = MemoryCache.Default.Select(kvp => kvp.Key).ToList();

        foreach (var cacheKey in cacheKeys)
        {
            await _logService.Add(LogEntryLevel.Trace, $"Memory cache removed for {cacheKey}");
            MemoryCache.Default.Remove(cacheKey);
        }
    }
}
     
