using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Reclaim.Api.Model;
using System.ComponentModel;
using System.Dynamic;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class StatusController
{
    private readonly Services.CacheService _cacheService;

    public StatusController(Services.CacheService cacheService)
    {
        _cacheService = cacheService;
    }
    
    /// <summary>
    /// Clear the cache
    /// </summary>
    /// <remarks></remarks>
    [HttpDelete("status/cache")]
    [RequireRole(Role.Administrator)]
    public async Task ClearCache()
    {
        await _cacheService.RemoveAll();
    }

    /// <summary>
    /// Generate an unhandled exception
    /// </summary>
    /// <remarks>
    /// ErrorCode.Unhandled
    /// </remarks>
    [HttpGet("status/error")]
    [AllowAnonymous]
    public void Error()
    {
        var i = 0;
        var j = 1 / i;
    }

    /// <summary>
    /// Return an error code, used to automatically generate all error codes for the frontend
    /// </summary>
    /// <remarks></remarks>
    [HttpPost("status/errorcodes")]
    [RequireRole(Role.Administrator)]
    public ErrorCode ErrorCodes()
    {
        return Model.ErrorCode.Unknown;
    }

    /// <summary>
    /// Determine if the API is alive
    /// </summary>
    /// <remarks></remarks>
    [HttpGet("status/ping")]
    [AllowAnonymous]
    public bool Ping()
    {
        return true;
    }
}