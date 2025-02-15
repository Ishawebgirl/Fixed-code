using Reclaim.Api.Model;

namespace Reclaim.Api.Services;

public abstract class BaseService
{
	protected (int, Role) GetCurrentAccountInfo(IHttpContextAccessor httpContextAccessor)
	{
		var claims = httpContextAccessor.HttpContext.User.Claims;
		var accountIDString = claims.Where(c => c.Type == "AccountID").FirstOrDefault();
		var roleString = claims.Where(c => c.Type == "Role").FirstOrDefault();

		if (accountIDString == null)
			throw new ApiException(ErrorCode.JwtClaimInvalid);

		var accountIDStringDecrypted = TwoWayEncryption.Decrypt(accountIDString.Value);

		int accountID;
		if (!int.TryParse(accountIDStringDecrypted, out accountID))
			throw new ApiException(ErrorCode.JwtClaimInvalid);

		Role role;
		if (!Enum.TryParse(roleString.Value, out role))
			throw new ApiException(ErrorCode.JwtClaimInvalid);

		return (accountID, role);
	}
}