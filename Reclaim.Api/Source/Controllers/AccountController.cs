using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.IdentityModel.Tokens;
using Reclaim.Api.Dtos;
using Reclaim.Api.Model;
using Reclaim.Api.Services;
using System.IdentityModel.Tokens.Jwt;
using System.Text;
using Security = System.Security.Claims;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class AccountController : BaseController
{
    private readonly SecurityService _securityService;
    private readonly CacheService _cacheService;
    private readonly ClaimService _claimService;
    private readonly DocumentService _documentService;
    private readonly LogService _logService;
    
    public AccountController(SecurityService securityService, CacheService cacheService, ClaimService claimService, DocumentService documentService, LogService logService)
    {
        _securityService = securityService;
        _cacheService = cacheService;
        _claimService = claimService;
        _documentService = documentService;
        _logService = logService;
    }

    /// <summary>
    /// Authenticate and receive a JWT bearer token    
    /// </summary>    
    /// <remarks>
    /// This endpoint returns a JWT bearer token that the caller may use to authorize requests to other API endpoints.  
    /// 
    /// A token received from this endpoint should be added to the request header as:
    /// 
    ///     Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjZmNzI1NDEwMW ...
    /// 
    /// This Swagger document will automatically add the required authorization header to subsequent calls following a successful authentication.
    /// 
    /// The refesh token GUID returned by this endpoint may be used to generate a new access token by calling the /accounts/authenticate/refresh endpoint with a valid access token in the header, and a valid refresh token in the body.
    ///
    /// ErrorCode.AccountCredentialsInvalid\
    /// ErrorCode.AccountTombstoned\
    /// ErrorCode.AccountLockedOutOverride\
    /// ErrorCode.AccountLockedOut\
    /// ErrorCode.AccountCredentialsExpired\
    /// ErrorCode.AccountEmailAddressNotConfirmed\
    /// ErrorCode.AccountRequiresIdentityProviderLocal
    /// </remarks>
    [HttpPost("accounts/authenticate")]
    [EnableRateLimiting("authentication")]
    [AllowAnonymous]
    public async Task<AuthenticationToken> Authenticate([FromBody] Dtos.AccountAuthentication authentication)
    {
        // await _logService.Add(LogEntryLevel.Trace, $"AUTHENTICATE on {this.HttpContext.Request.Host}");

        var account = null as Model.Account;
        var emailAddresses = authentication.EmailAddress.Split(':');
        var ip = Request.HttpContext.Connection.RemoteIpAddress?.ToString();

        if (emailAddresses.Length == 2)
            account = await _securityService.AuthenticateAndImpersonate(emailAddresses[0], authentication.Password, ip, emailAddresses[1]);
        else
            account = await _securityService.Authenticate(authentication.EmailAddress, authentication.Password, ip);

        if (account == null)
            throw new ApiException(ErrorCode.AccountCredentialsInvalid, $"Failed to authenticate as {authentication.EmailAddress}, unknown error.");

        var authenticationToken = await CreateToken(account);

        SetAuthenticationCookie(authenticationToken);

        return authenticationToken;
    }

    /// <summary>
    /// Authorize access to API resources by authenticating to Google using OAuth 2.0 
    /// </summary>
    /// <remarks>
    /// This Swagger document will automatically add the required authorization header to subsequent calls following a successful authentication.
    ///
    /// ErrorCode.GoogleJwtBearerTokenInvalid\
    /// ErrorCode.AccountExternalCredentialsInvalid\
    /// ErrorCode.AccountLockedOutOverride\
    /// ErrorCode.AccountLockedOut\
    /// ErrorCode.AccountCredentialsExpired\
    /// ErrorCode.AccountEmailAddressNotConfirmed\
    /// ErrorCode.AccountRequiresIdentityProviderGoogle
    /// </remarks>
    [HttpPost("accounts/authenticate/google")]
    [EnableRateLimiting("authentication")]
    [AllowAnonymous]
    public async Task<AuthenticationToken> AuthenticateGoogle([FromBody] Dtos.GoogleAccountAuthentication authentication)
    {
        var ip = Request.HttpContext.Connection.RemoteIpAddress?.ToString();

        var decodedGoogleCredential = _securityService.ValidateGoogleCredential(authentication.GoogleJwt, authentication.EmailAddress);

        if (decodedGoogleCredential == null)
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"Failed to authenticate via Google, unable to decode Google JWT {authentication.GoogleJwt}.");

        var account = await _securityService.AuthenticateExternal(decodedGoogleCredential.EmailAddress, IdentityProvider.Google, ip);

        var authenticationToken = await CreateToken(account);

        SetAuthenticationCookie(authenticationToken);

        return authenticationToken;
    }

    /// <summary>
    /// Reauthenticate to the local database using a refresh token instead of a password
    /// </summary>
    /// <remarks>
    /// This endpoint returns a JWT bearer token that a caller may use to authorize requests to other API endpoints, but unlike /accounts/authenticate, this endpoint does not accept a password but rather a refresh token.
    /// 
    /// A token received from this endpoint should be added to the request header as normal.
    /// 
    /// This Swagger document will automatically add the required authorization header to subsequent calls following a successful authentication.
    /// 
    /// ErrorCode.JwtBearerTokenInvalid\
    /// ErrorCode.AccountDoesNotExist\
    /// ErrorCode.AccountStatusInvalidForOperation
    /// </remarks>
    [HttpPost("accounts/authenticate/refresh")]
    [EnableRateLimiting("authentication")]
    [RequireRole]
    public async Task<AuthenticationToken> AuthenticateRefresh([FromBody] Dtos.AccountAuthenticationRefresh authentication)
    {
        var error = "Failed to refresh authentication token";
        var emailAddress = authentication.EmailAddress.Split(':').First();
        var ip = Request.HttpContext.Connection.RemoteIpAddress?.ToString();

        if (this.Current.EmailAddress != emailAddress)
            throw new ApiException(ErrorCode.JwtBearerTokenInvalid, $"{error}, bearer token is invalid.");

        if (authentication.RefreshToken == "_") // this is a request with a server-based cookie coming from the website
            authentication.RefreshToken = ServerSideCookie.GetValue(this.HttpContext, CookieName.RefreshToken);

        if (this.Current.RefreshToken != authentication.RefreshToken)
            throw new ApiException(ErrorCode.JwtRefreshTokenInvalid, $"{error}, refresh token is invalid.");

        var account = await _securityService.GetAccount(emailAddress);

        if (account == null)
            throw new ApiException(ErrorCode.AccountDoesNotExist, $"{error}, no account with the current JTW token claim data exists.");
        if (account.Status != AccountStatus.Normal)
            throw new ApiException(ErrorCode.AccountStatusInvalidForOperation, $"{error}, account status is {account.Status}.");

        await _securityService.RecordAccessTokenRefresh(account.ID, ip);

        var authenticationToken = await CreateToken(account);

        SetAuthenticationCookie(authenticationToken);

        return authenticationToken;
    }

    /// <summary>
    /// Retrieve a specific account by email address
    /// </summary>
    /// <remarks>
    /// ErrorCode.AccountDoesNotExist
    /// </remarks>
    /// <param name="emailAddress" example="sample@test.com">The account's email address</param>
    [HttpGet("accounts")]
    [RequireRole(Role.Administrator)]
    public async Task<Dtos.Account> GetByEmailAddress([FromQuery] string emailAddress)
    {
        var account = await _securityService.GetAccount(emailAddress);

        if (account == null)
            throw new ApiException(ErrorCode.AccountDoesNotExist, $"No account with email address {emailAddress} exists.");

        return new Dtos.Account(account);
    }

    /// <summary>
    /// View a list of currently signed-in accounts
    /// </summary>
    [HttpGet("accounts/authenticated")]
    [RequireRole(Role.Administrator)]
    public async Task<List<Dtos.Account>> Authenticated()
    {
        var accounts = await _securityService.GetAuthenticatedAccounts();

        return accounts.Select(x => new Dtos.Account(x)).ToList();
    }

    /// <summary>
    /// Retrieve a specific account by unique id
    /// </summary>
    /// <remarks>
    /// ErrorCode.AccountDoesNotExist
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The account's public unique ID</param>
    [HttpGet("accounts/{uniqueID}")]
    [RequireRole(Role.Administrator)]
    public async Task<Dtos.Account> Get([FromRoute] Guid uniqueID)
    {
        var account = await _securityService.GetAccount(uniqueID);

        if (account == null)
            throw new ApiException(ErrorCode.AccountDoesNotExist, $"No account with UniqueID {uniqueID} exists.");

        return new Dtos.Account(account);
    }

    /// <summary>
    /// Retrieve all claims in the system, currently not paged or limited
    /// </summary>
    [HttpGet("accounts/administrators")]
    [RequireRole(Role.Administrator)]
    public async Task<List<Dtos.Administrator>> GetAdministrators()
    {
        var administrators = await _securityService.GetAdministrators();

        return administrators.Select(x => new Dtos.Administrator(x)).ToList();
    }

    /// <summary>
    /// Confirm an account via the welcome email workflow
    /// </summary>
    /// <param name="dto" example="">An account confirmation DTO</param>
    /// <remarks>
    /// ErrorCode.AccountCredentialsInvalid\
    /// ErrorCode.AccountAlreadyConfirmed\
    /// ErrorCode.AccountAlreadyConfirmed\
    /// ErrorCode.AccountMagicUrlTokenInvalid\
    /// ErrorCode.AccountMagicUrlTokenExpired
    /// </remarks>
    [HttpPost("accounts/confirm")]
    [AllowAnonymous]
    public async Task Confirm([FromBody] AccountConfirmation dto)
    {
        try
        {
            var account = await _securityService.Confirm(dto.EmailAddress, Guid.Parse(dto.Token));
        }
        catch (ApiException ex)
        {
            // hide informative messages to avoid dictionary attacks
            switch (ex.ErrorCode)
            {
                case ErrorCode.AccountAlreadyConfirmed:
                case ErrorCode.AccountMagicUrlTokenExpired:
                    throw;

                default:
                    await _logService.Add(ex);
                    break;
            }
        }
    }

    /// <summary>
    /// Retrieve the account associated with the current JWT access token
    /// </summary>
    /// <remarks>
    /// ErrorCode.AccountDoesNotExist
    /// </remarks>
    [HttpGet("accounts/me")]
    [RequireRole]
    public async Task<Dtos.Account> Me()
    {
        var account = await _securityService.GetAccount(Current.AccountID);
        return new Dtos.Account(account);
    }

    /// <summary>
    /// Accept magic URL token and new password for password reset workflow
    /// </summary>
    /// <param name="dto" example="">A PasswordReset DTO</param>
    /// <remarks>
    /// ErrorCode.AccountDoesNotExist\
    /// ErrorCode.AccountRoleInvalidForOperation\
    /// ErrorCode.AccountRequiresIdentityProviderLocal\
    /// ErrorCode.AccountStatusInvalidForOperation\
    /// ErrorCode.AccountMagicUrlTokenInvalid\
    /// ErrorCode.AccountMagicUrlTokenExpired\
    /// ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity\
    /// ErrorCode.AccountPasswordUsedPreviously
    /// </remarks>
    [EnableRateLimiting("authentication")]
    [HttpPut("accounts/password")]
    [AllowAnonymous]
    public async Task ResetPassword([FromBody] Dtos.PasswordReset dto)
    {
        try
        {
            var account = await _securityService.ResetPassword(dto.EmailAddress, dto.NewPassword, Guid.Parse(dto.Token));
        }
        catch (ApiException ex)
        {
            switch (ex.ErrorCode)
            {
                case ErrorCode.AccountMagicUrlTokenExpired:
                case ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity:
                case ErrorCode.AccountPasswordUsedPreviously:
                    throw;

                default:
                    await _logService.Add(ex);
                    break;
            }
        }
    }

    /// <summary>
    /// Request an URL to be sent to the email address on record, allowing a password reset
    /// </summary>
    /// <param name="dto" example="">A PasswordResetRequest dto containing the account's email address</param>
    /// <remarks>
    /// ErrorCode.AccountDoesNotExist\
    /// ErrorCode.AccountRoleInvalidForOperation\
    /// ErrorCode.AccountRequiresIdentityProviderLocal\
    /// ErrorCode.AccountRoleInvalidForOperation
    /// </remarks>        
    [EnableRateLimiting("authentication")]
    [HttpPost("accounts/password/reset")]
    [AllowAnonymous]
    public async Task RequestResetPassword([FromBody] Dtos.PasswordResetRequest dto)
    {
        try
        {
            var account = await _securityService.RequestPasswordReset(dto.EmailAddress);
        }
        catch (Exception ex)
        {
            await _logService.Add(ex);
        }
    }

    /// <summary>
    /// Upload the account's avatar image
    /// </summary>
    /// <remarks>
    /// ErrorCode.DocumentUploadToAzureFailed\
    /// ErrorCode.DocumentTypeNotSupported  
    /// </remarks>
    [HttpPost("accounts/avatar")]
    [RequireRole]
    public async Task<string> UploadAvatar(IFormFile file)
    {
        var url = await _documentService.UploadAvatar(file);

        return url;
    }

    private void SetAuthenticationCookie(AuthenticationToken token)
    {
        var cookieOptions = new CookieOptions
        {
            HttpOnly = true,
            Secure = true,
            Expires = DateTime.UtcNow.AddSeconds(Setting.JwtAccessTokenTimeout),
            SameSite = SameSiteMode.None,
            Domain = Setting.CookieDomain,
            Path = "/"
        };

        this.HttpContext.Response.Cookies.Append(CookieName.AccessToken, token.AccessToken, cookieOptions);
        this.HttpContext.Response.Cookies.Append(CookieName.RefreshToken, token.RefreshToken, cookieOptions);
    }

    private async Task<AuthenticationToken> CreateToken(Model.Account account)
    {
        var jti = Guid.NewGuid().ToString();
        var refreshToken = Guid.NewGuid().ToString();
        var now = DateTime.UtcNow;
        var validUntil = now.AddSeconds(Setting.JwtAccessTokenTimeout);

        var authClaims = new List<Security.Claim>
        {
            new Security.Claim(JwtRegisteredClaimNames.Jti, jti),
            new Security.Claim("AccountID", TwoWayEncryption.Encrypt(account.ID.ToString())), // this is gross, but I don't want to have a visible account ID in the claims list, I could pass the unique ID, but then every time I write a log entry, I'll have to look up the account PK 
            new Security.Claim("EmailAddress", account.EmailAddress),
            new Security.Claim("Role", account.Role.ToString()),
            new Security.Claim("RefreshToken", refreshToken)
        };

        var authSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(Setting.JwtAccessTokenSecret));

        var token = new JwtSecurityToken(
            issuer: Setting.ApiRootUrl,
            notBefore: now,
            expires: validUntil,
            claims: authClaims,
            signingCredentials: new SigningCredentials(authSigningKey, SecurityAlgorithms.HmacSha256)
        );

        var accessToken = new JwtSecurityTokenHandler().WriteToken(token);

        await _logService.Add(LogEntryLevel.Trace, $"Saving JTI {jti} from {HttpContext.Request.Host}");

        await _cacheService.Set($"JTI-{account.EmailAddress}", jti, Setting.JwtAccessTokenTimeout, true);

        return new AuthenticationToken { AccessToken = accessToken, RefreshToken = refreshToken, ValidUntil = validUntil, EmailAddress = account.EmailAddress, Role = account.Role, NiceName = account.NiceName, AvatarUrl = account.AvatarUrl };
    }
}
