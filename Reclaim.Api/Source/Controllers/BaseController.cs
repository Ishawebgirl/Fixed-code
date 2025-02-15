using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure.Internal;
using Reclaim.Api.Model;
using Reclaim.Api.Services;

namespace Reclaim.Api.Controllers;

[RequireRole]
public abstract class BaseController : Controller
{
    public class AuthenticatedAccount
    {
        public int AccountID { get; set; }
        public string EmailAddress { get; set; }
        public Role Role { get; set; }
        public string RefreshToken { get; set; }
    }

    private AuthenticatedAccount? _current = null;
    
    public BaseController() : base()
    {        
    }

    protected void Prevalidate()
    {
        if (!ModelState.IsValid)
            throw new ApiException(ErrorCode.ModelValidationFailed, string.Join(", ", ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage)));
    }

    protected AuthenticatedAccount Current
    {
        get
        {
            if (_current != null)
                return _current;
            else
            {
                // required claims
                var claims = HttpContext.User.Claims;
                var accountIDString = claims.Where(c => c.Type == "AccountID").FirstOrDefault();
                var emailAddress = claims.Where(c => c.Type == "EmailAddress").FirstOrDefault();
                var roleString = claims.Where(c => c.Type == "Role").FirstOrDefault();
                var refreshTokenString = claims.Where(c => c.Type == "RefreshToken").FirstOrDefault();

                if (accountIDString == null || emailAddress == null || roleString == null || refreshTokenString == null)
                    throw new ApiException(ErrorCode.JwtClaimInvalid);

                var accountIDStringDecrypted = TwoWayEncryption.Decrypt(accountIDString.Value);

                int accountID;
                if (!int.TryParse(accountIDStringDecrypted, out accountID))
                    throw new ApiException(ErrorCode.JwtClaimInvalid);

                Role role;
                if (!Enum.TryParse(roleString.Value, out role))
                    throw new ApiException(ErrorCode.JwtClaimInvalid);

                _current = new AuthenticatedAccount
                {
                    AccountID = accountID,
                    EmailAddress = emailAddress.Value,
                    Role = role,
                    RefreshToken = refreshTokenString.Value
                };

                return _current;
            }
        }
    }
}

