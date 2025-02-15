using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.IdentityModel.Tokens;
using Reclaim.Api.Model;
using Reclaim.Api.Services;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

namespace Reclaim.Api;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true, Inherited = true)]

public class RequireRoleAttribute : ActionFilterAttribute, IAsyncActionFilter
{
    private readonly Role[] Roles;
    private SecurityService _securityService;
    private CacheService _cacheService;
    private LogService _logService;
    private ActionExecutingContext _context;
    
    public RequireRoleAttribute()
    {
        this.Roles = new Role[0];
    }

    public RequireRoleAttribute(Model.Role requiresRole)
    {
        this.Roles = new Role[] { requiresRole };
    }

    public RequireRoleAttribute(Role[] requiresRole)
    {
        this.Roles = requiresRole;
    }

    public async override Task OnActionExecutionAsync(ActionExecutingContext context, ActionExecutionDelegate next)
    {
        _context = context;

        var services = context.HttpContext.RequestServices;
        _securityService = services.GetService<SecurityService>();
        _cacheService = services.GetService<CacheService>();
        _logService = services.GetService<LogService>();

        var allowAnonymous = context.ActionDescriptor.EndpointMetadata.OfType<AllowAnonymousAttribute>().Any();
        
        if (allowAnonymous)
        {
            await next();
            return;
        }

        if (!context.HttpContext.Request.Headers.TryGetValue("Authorization", out var authorizationHeader))
            throw new ApiException(ErrorCode.JwtBearerTokenMissing, "Authorization header is missing.");

        var token = DecodeAndValidateToken(authorizationHeader);
        var account = await ValidateClaims(token);

        ValidateRole(token, account);

        _securityService.SetLastActiveTimestampSync(account.ID);

        var identity = new ClaimsIdentity(token.Claims);
        var principal = new ClaimsPrincipal(identity);

        context.HttpContext.User = principal;
        await next();
    }

    public Role[] GetRoles()
    {
       return this.Roles;
    }

    private JwtSecurityToken DecodeAndValidateToken(string header)
    {
        if (!header.StartsWith("Bearer "))
            throw new ApiException(ErrorCode.JwtBearerTokenMissing, "Bearer token is missing.");

        var tokenString = header.Right(header.Length - 7);
        var authSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(Setting.JwtAccessTokenSecret));

        var validationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = Setting.ApiRootUrl,
            ValidateAudience = false,
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = authSigningKey,
            ValidateLifetime = true
        };

        var validatedToken = null as SecurityToken;

        try
        {
            new JwtSecurityTokenHandler().ValidateToken(tokenString, validationParameters, out validatedToken);
        }
        catch (Exception ex)
        {
            throw new ApiException(ErrorCode.JwtBearerTokenInvalid, $"Bearer token is invalid. {ex.Message}");
        }

        if (validatedToken == null)
            throw new ApiException(ErrorCode.JwtBearerTokenInvalid, $"Bearer token is invalid.");

        var token = (JwtSecurityToken)validatedToken!;

        return token;
    }

    private async Task<Account> ValidateClaims(JwtSecurityToken token)
    {
        var invalid = "Bearer token is invalid.";
        var expired = "Bearer token is expired.";
        var emailAddress = token.Claims.FirstOrDefault(x => x.Type == "EmailAddress")?.Value;

        if (emailAddress == null)
            throw new ApiException(ErrorCode.JwtClaimNotPresent, invalid);

        var account = await _securityService.GetAccount(emailAddress);

        if (account == null)
            throw new ApiException(ErrorCode.JwtClaimNotPresent, invalid);

        await _logService.Add(LogEntryLevel.Trace, $"Reading JTI from cache on {_context.HttpContext.Request.Host}");

        var cachedJti = await _cacheService.Get<string>($"JTI-{account.EmailAddress}", Setting.JwtAccessTokenTimeout, null, true);

        if (cachedJti == null)
            throw new ApiException(ErrorCode.JwtBearerTokenExpired, expired);

        var jti = token.Claims.FirstOrDefault(x => x.Type == JwtRegisteredClaimNames.Jti)?.Value ?? "";
        await _logService.Add(LogEntryLevel.Trace, $"Comparing JTI from {_context.HttpContext.Request.Host}, token JTI:{jti}, cached JTI: {cachedJti}");

        if (jti != cachedJti)
            throw new ApiException(ErrorCode.JwtBearerTokenInvalid, invalid);

        var accountID = token.Claims.FirstOrDefault(x => x.Type == "AccountID")?.Value;
        var accountIDDecrypted = TwoWayEncryption.Decrypt(accountID);

        if (account.ID.ToString() != accountIDDecrypted)
            throw new ApiException(ErrorCode.JwtBearerTokenInvalid, invalid);

        return account;
    }

    private void ValidateRole(JwtSecurityToken token, Account account)
    {
        var error = "Bearer token is invalid.";

        if (this.Roles == null || this.Roles.Length == 0)
            return;

        var roleString = token.Claims.Where(x => x.Type == "Role").FirstOrDefault();
        Role role;

        if (!Enum.TryParse(roleString.Value, out role))
            throw new ApiException(ErrorCode.JwtClaimNotPresent, error);

        if (role != account.Role)
            throw new ApiException(ErrorCode.JwtClaimInvalid, error);

        if (!this.Roles.Contains(role))
            throw new ApiException(ErrorCode.JwtRoleInvalid, "Account role invalid for this operation");

    }
}
