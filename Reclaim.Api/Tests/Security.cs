using RestSharp;
using Newtonsoft.Json;
using RestSharp.Authenticators;
using System;
using Newtonsoft.Json.Linq;
using System.Dynamic;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Reclaim.Api.Tests;

[Collection("Sequential")]
public class Security : Base
{
    public Security()
    {
        Database.Reset();
        
        _restClient = new RestClient(Default.Url);
    }

    [Fact]
    public void AuthenticationHappyPath()
    {
        var data = GetJwtAccessToken();
        Assert.Equal(200, data.statusCode);
        Assert.True(data.accessToken.Length > 100);

        var expiresIn = (data.validUntil - DateTime.UtcNow).TotalSeconds;
        Assert.True(expiresIn > 500 && expiresIn < 600);
        Assert.True(Guid.TryParse(data.refreshToken.ToString(), out Guid dummy));

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.PasswordExpiredTimestamp == null);
        Assert.True(account.EmailAddressConfirmedTimestamp != null);
        Assert.True(account.TombstonedTimestamp == null);
        Assert.True(account.LockedOutTimestamp == null);
        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.AuthenticatedTimestamp == account.SessionAuthenticatedTimestamp);
        Assert.True(account.FailedAuthenticationCount == 0);

        var authentications = Database.GetAuthentications(Default.CustomerEmailAddress);
        Assert.True(authentications.Count() == 1);

        var authentication = authentications.Last();
        Assert.True(authentication.IsSuccessful);
        Assert.True(!authentication.IsRefresh);
        Assert.True(!authentication.IsShadowed);
    }

    [Fact]
    public void AuthenticationRefreshHappyPath()
    {
        var data = GetJwtAccessToken();
        var refreshData = GetJwtAccessToken(data.emailAddress, refreshToken: data.refreshToken, accessToken: data.accessToken);
        Assert.True(refreshData.statusCode == 200);
        Assert.True(refreshData.accessToken.Length > 100);

        var expiresIn = (refreshData.validUntil - DateTime.UtcNow).TotalSeconds;
        Assert.True(expiresIn > 500 && expiresIn < 600);
        Assert.True(Guid.TryParse(data.refreshToken.ToString(), out Guid dummy));

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.PasswordExpiredTimestamp == null);
        Assert.True(account.EmailAddressConfirmedTimestamp != null);
        Assert.True(account.TombstonedTimestamp == null);
        Assert.True(account.LockedOutTimestamp == null);
        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.SessionAuthenticatedTimestamp != null);
        Assert.True(account.AuthenticatedTimestamp != account.SessionAuthenticatedTimestamp);
        Assert.True(account.FailedAuthenticationCount == 0);

        var authentications = Database.GetAuthentications(Default.CustomerEmailAddress);
        Assert.True(authentications.Count() == 2);

        var authentication = authentications.Last();
        Assert.True(authentication.IsSuccessful);
        Assert.True(authentication.IsRefresh);
        Assert.True(!authentication.IsShadowed);
    }

    [Fact]
    public void AuthenticationRefreshBadAccessToken()
    {
        var data = GetJwtAccessToken();

        var refreshData = GetJwtAccessToken(data.emailAddress, refreshToken: data.refreshToken, accessToken: "assfsdlkjsfdljk");
        Assert.True(refreshData.statusCode == 401);
        Assert.True(refreshData.errorCodeName == "JwtBearerTokenInvalid");
    }

    [Fact]
    public void AuthenticationRefreshNoRefreshToken()
    {
        var data = GetJwtAccessToken();

        var refreshData = GetJwtAccessToken(data.emailAddress, refreshToken: "", accessToken: data.accessToken);
        Assert.True(refreshData.statusCode == 400);
        Assert.True(refreshData.errorCodeName == "ModelValidationFailed");
        Assert.True(refreshData.message.Contains("RefreshToken cannot be empty"));
    }

    [Fact]
    public void AuthenticationRefreshInvalidToken()
    {
        var data = GetJwtAccessToken();

        var refreshData = GetJwtAccessToken(data.emailAddress, refreshToken: Guid.NewGuid().ToString(), accessToken: data.accessToken);
        Assert.True(refreshData.statusCode == 401);
        Assert.True(refreshData.errorCodeName == "JwtRefreshTokenInvalid");
    }

    [Fact]
    public void AuthenticationRefreshInvalidClaim()
    {
        var data = GetJwtAccessToken();

        var refreshData = GetJwtAccessToken(Default.InvalidEmailAddress, refreshToken: data.refreshToken, accessToken: data.accessToken);
        Assert.True(refreshData.statusCode == 401);
        Assert.True(refreshData.errorCodeName == "JwtBearerTokenInvalid");
    }

    [Fact]
    public void InvalidUsername()
    {
        var data = GetJwtAccessToken(Default.InvalidEmailAddress);
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountCredentialsInvalid");
        Assert.True(data.message.StartsWith("Failed to authenticate"));

        var authentications = Database.GetAuthentications(Default.CustomerEmailAddress);
        Assert.True(authentications.Count() == 0);

        var lastAuthentication = Database.GetLastAuthentications(1).First();
        Assert.True(lastAuthentication.AccountID == null);
    }

    [Fact]
    public void InvalidPassword()
    {
        var data = GetJwtAccessToken(Default.CustomerEmailAddress, "xxx");
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountCredentialsInvalid");
        Assert.True(data.message.StartsWith("Failed to authenticate"));

        var authentications = Database.GetAuthentications(Default.CustomerEmailAddress);
        Assert.True(authentications.Count() == 1);

        var authentication = authentications.Last();
        Assert.True(!authentication.IsSuccessful);
    }

    [Fact]
    public void BasicRequest()
    {
        dynamic data = RequestWithJwtAccessToken("accounts/me", Method.Get, null);
        Assert.True(data.statusCode == 200);
        Assert.True(data.emailAddress == Default.CustomerEmailAddress);
        Assert.True(data.authenticatedTimestamp != null);
        Assert.True(data.authenticatedTimestamp == data.sessionAuthenticatedTimestamp);
        Assert.True(data.role == "Customer");
        Assert.True(data.identityProvider == "Local");
    }

    [Fact]
    public void NoAccessToken()
    {
        var request = new RestRequest("accounts/me", Method.Get);
        var response = _restClient.Execute(request);
        var json = response.Content ?? "";

        dynamic data = JsonConvert.DeserializeObject<ExpandoObject>(json);
        Assert.True((int)response.StatusCode == 401);
        Assert.True(data!.errorCodeName == "JwtBearerTokenMissing");
    }

    [Fact]
    public void ValidRole()
    {
        dynamic data = RequestWithJwtAccessToken($"administrator/accounts/{Default.CustomerAccountUniqueID}", Method.Get, null, Default.AdministratorEmailAddress, Default.AdministratorPassword);
        Assert.True(data.statusCode == 200);
        Assert.True(data.emailAddress == Default.CustomerEmailAddress);
        Assert.True(data.role == "Customer");
    }
    
    [Fact]
    public void InvalidRole()
    {
        dynamic data = RequestWithJwtAccessToken($"administrator/accounts/{Default.CustomerAccountUniqueID}", Method.Get, null, Default.CustomerEmailAddress, Default.CustomerPassword);
        Assert.True(data.statusCode == 403);
        Assert.True(data.errorCodeName == "JwtRoleInvalid");
    }

    [Fact]
    public void InvalidAccessToken()
    {
        var token = GetJwtAccessToken();

        var request = new RestRequest("accounts/me", Method.Get);
        request.AddHeader("Authorization", $"Bearer x{token.accessToken}");

        var response = _restClient.Execute(request);
        var json = response.Content ?? "";

        dynamic data = JsonConvert.DeserializeObject<ExpandoObject>(json);
        Assert.True((int)response.StatusCode == 401);
        Assert.True(data!.errorCodeName == "JwtBearerTokenInvalid");
    }

    [Fact]
    public void LockedOutOverride()
    {
        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        
        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountLockedOutOverride");
    }

    [Fact]
    public void LockedOut()
    {
        for (var i = 0; i < 10; i++)
            GetJwtAccessToken(Default.CustomerEmailAddress, "password");

        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountLockedOut");
    }

    [Fact]
    public void LockedOutAfter10Minutes()
    {
        LockedOut();

        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow.AddMinutes(-10));

        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountLockedOut");

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.LockedOutTimestamp != null);
        Assert.True(account.FailedAuthenticationCount == 10);
    }

    [Fact]
    public void LockedOutAfter40Minutes()
    {
        LockedOut();

        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow.AddMinutes(-40));

        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 200);

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.LockedOutTimestamp == null);
        Assert.True(account.FailedAuthenticationCount == 0);
    }

    [Fact]
    public void Tombstoned()
    {
        Database.SetAccountTombstonedTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountTombstoned");
    }

    [Fact]
    public void PasswordExpired()
    {
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        
        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountCredentialsExpired");
    }

    [Fact]
    public void RequestPasswordReset()
    {
        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress
        };

        var data = RequestWithoutJwtAccessToken("accounts/password/reset", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
    }

    [Fact]
    public void RequestPasswordResetInvalidAccount()
    {
        var parameters = new
        {
            emailAddress = Default.InvalidEmailAddress
        };

        var data = RequestWithoutJwtAccessToken("accounts/password/reset", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountDoesNotExist", "the account does not exist");
    }

    [Fact]
    public void RequestPasswordResetInvalidAccountRole()
    {
        var parameters = new
        {
            emailAddress = Default.AdministratorEmailAddress
        };

        var data = RequestWithoutJwtAccessToken("accounts/password/reset", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountRoleInvalidForOperation");
    }

    [Fact]
    public void RequestPasswordResetInvalidForTombstoned()
    {
        Database.SetAccountTombstonedTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress
        };

        var data = RequestWithoutJwtAccessToken("accounts/password/reset", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountStatusInvalidForOperation");
    }

    [Fact]
    public void RequestPasswordResetInvalidForLockedOut()
    {
        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress
        };

        var data = RequestWithoutJwtAccessToken("accounts/password/reset", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountStatusInvalidForOperation");
    }

    [Fact]
    public void PasswordReset()
    {
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        var before = Database.GetAccount(Default.CustomerEmailAddress);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerNewPassword,
            token = Default.CustomerMagicUrlToken
        };

        var data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.SessionAuthenticatedTimestamp != null);
        Assert.True(account.EmailAddressConfirmedTimestamp != null);
        Assert.True(account.MagicUrlToken == null);
        Assert.True(account.MagicUrlValidUntil == null);
        Assert.True(account.PasswordChangedTimestamp != null);
    }

    [Fact]
    public void PasswordResetValidations()
    {
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        Database.ClearAccountPasswordChangedTimestamp(Default.CustomerEmailAddress);

        var parameters = new
        {
            emailAddress = " ",
            newPassword = Default.CustomerPassword,
            token = Default.CustomerMagicUrlToken
        };

        var data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "ModelValidationFailed");
        Assert.True(data.message.IndexOf("The entity failed at least one validation test: EmailAddress cannot be all whitespace") >= 0);

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = " ",
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "ModelValidationFailed");
        Assert.True(data.errorCodeName == "ModelValidationFailed");
        Assert.True(data.message.IndexOf("The entity failed at least one validation test: NewPassword cannot be all whitespace") >= 0);

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerPassword,
            token = " "
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "ModelValidationFailed");
        Assert.True(data.message.IndexOf("The entity failed at least one validation test: Token cannot be all whitespace") >= 0);

        parameters = new
        {
            emailAddress = Default.InvalidEmailAddress,
            newPassword = Default.CustomerNewPassword,
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountDoesNotExist", "the account does not exist");

        parameters = new
        {
            emailAddress = Default.AdministratorEmailAddress,
            newPassword = Default.AdministratorPassword,
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountRoleInvalidForOperation", "an account with role");

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerNewPassword,
            token = "58FBA86D-11F8-4976-A092-D6FBC2F0676C"
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountMagicUrlTokenInvalid", "the magic URL token provided in the request does not match the stored value");

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.SessionAuthenticatedTimestamp != null);
        Assert.True(account.PasswordExpiredTimestamp != null);
        Assert.True(account.PasswordChangedTimestamp == null);
        Assert.True(account.MagicUrlToken != null);
        Assert.True(account.MagicUrlValidUntil != null);

        Database.Reset();
        
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        Database.SetAccountTombstonedTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerPassword,
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountStatusInvalidForOperation", "an account with status Tombstoned");

        Database.Reset();
        
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerNewPassword,
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatLastLogEntryIs("AccountStatusInvalidForOperation", "an account with status LockedOut");

        Database.Reset();
        
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(-10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerNewPassword,
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "AccountMagicUrlTokenExpired");
        Assert.True(data.message.IndexOf("the magic URL token has expired") >= 0);

        Database.Reset();
        
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = "password",
            token = Default.CustomerMagicUrlToken
        };

        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "AccountPasswordDoesNotMeetMinimumComplexity");
        Assert.True(data.message.IndexOf("the password does not meet minimum complexity rules") >= 0);
    }

    [Fact]
    public void PasswordResetNotUnique()
    {
        var before = Database.GetAccount(Default.CustomerEmailAddress);

        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            newPassword = Default.CustomerNewPassword,
            token = Default.CustomerMagicUrlToken
        };

        var data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 200);

        var account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.SessionAuthenticatedTimestamp != null);
        Assert.True(account.PasswordExpiredTimestamp == null);
        Assert.True(account.PasswordChangedTimestamp != null);
        Assert.True(account.MagicUrlToken == null);
        Assert.True(account.MagicUrlValidUntil == null);

        var previousPasswords = Database.GetAccountPasswords(Default.CustomerEmailAddress);
        Assert.True(previousPasswords.Count() == 1);

        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

        // try with current password
        data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "AccountPasswordUsedPreviously");
        Assert.True(data.message.IndexOf("the password has been used previously") >= 0);

        account = Database.GetAccount(Default.CustomerEmailAddress);
        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.SessionAuthenticatedTimestamp != null);
        Assert.True(account.AuthenticatedTimestamp == account.SessionAuthenticatedTimestamp);
        Assert.True(account.PasswordExpiredTimestamp != null);
        Assert.True(account.PasswordChangedTimestamp != null);
        Assert.True(account.MagicUrlToken != null);
        Assert.True(account.MagicUrlValidUntil != null);

        previousPasswords = Database.GetAccountPasswords(Default.CustomerEmailAddress);
        Assert.True(previousPasswords.Count() == 1);

        // try multiple passwords, then pre-usesd
        var passwords = new List<string> { "9Dd92j7#", "*7@OL1ks", "45TyE0!k", Default.CustomerPassword };

        foreach (var password in passwords)
        {
            Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
            Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);

            parameters = new
            {
                emailAddress = Default.CustomerEmailAddress,
                newPassword = password,
                token = Default.CustomerMagicUrlToken
            };

            data = RequestWithoutJwtAccessToken("accounts/password", Method.Put, parameters);

            if (password == Default.CustomerPassword)
            {
                Assert.True(data.statusCode == 400);
                Assert.True(data.errorCodeName == "AccountPasswordUsedPreviously");
                Assert.True(data.message.IndexOf("the password has been used previously") >= 0);
            }
            else
            {
                Assert.True(data.statusCode == 200);
            }
        }

        previousPasswords = Database.GetAccountPasswords(Default.CustomerEmailAddress);
        Assert.True(previousPasswords.Count() == 4);
    }

    [Fact]
    public void EmailAddressNotConfirmed()
    {
        Database.ClearAccountEmailAddressConfirmedTimestamp(Default.CustomerEmailAddress);

        var data = GetJwtAccessToken();
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountEmailAddressNotConfirmed");
    }

    [Fact]
    public void EmailAddressConfirmation()
    {
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.ClearAccountEmailAddressConfirmedTimestamp(Default.CustomerEmailAddress);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            token = Default.CustomerMagicUrlToken
        };

        var data = RequestWithoutJwtAccessToken("accounts/confirm", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        // ? CustomAssert.ThatLastLogEntryIs(data, "emailAddress");
        
        var account = Database.GetAccount(Default.CustomerEmailAddress);

        Assert.True(account.AuthenticatedTimestamp != null);
        Assert.True(account.SessionAuthenticatedTimestamp != null);
        Assert.True(account.EmailAddressConfirmedTimestamp != null);
        Assert.True(account.MagicUrlToken == null);
        Assert.True(account.MagicUrlValidUntil == null);
    }

    [Fact]
    public void EmailAddressConfirmationFailedInvalidEmailAddress()
    {
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.ClearAccountEmailAddressConfirmedTimestamp(Default.CustomerEmailAddress);

        var parameters = new
        {
            emailAddress = Default.InvalidEmailAddress,
            token = Default.CustomerMagicUrlToken
        };

        var data = RequestWithoutJwtAccessToken("accounts/confirm", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatObjectDoesNotHaveProperty(data, "emailAddress"); 
    }

    [Fact]
    public void EmailAddressConfirmationFailedInvalidUniqueID()
    {
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.ClearAccountEmailAddressConfirmedTimestamp(Default.CustomerEmailAddress);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            token = "3359f7b0-36a2-43f0-9158-4dfde34f26b1"
        };

        var data = RequestWithoutJwtAccessToken("accounts/confirm", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatObjectDoesNotHaveProperty(data, "emailAddress"); 
    }

    [Fact]
    public void EmailAddressDoubleConfirmation()
    {
        Database.SetAccountMagicUrlToken(Default.CustomerEmailAddress, Default.CustomerMagicUrlToken, DateTime.UtcNow.AddMinutes(10));
        Database.ClearAccountEmailAddressConfirmedTimestamp(Default.CustomerEmailAddress);

        var parameters = new
        {
            emailAddress = Default.CustomerEmailAddress,
            token = Default.CustomerMagicUrlToken
        };

        var data = RequestWithoutJwtAccessToken("accounts/confirm", Method.Post, parameters);
        Assert.True(data.statusCode == 200);
        CustomAssert.ThatObjectDoesNotHaveProperty(data, "emailAddress"); 

        data = RequestWithoutJwtAccessToken("accounts/confirm", Method.Post, parameters);
        Assert.True(data.statusCode == 400);
        Assert.True(data.errorCodeName == "AccountAlreadyConfirmed");
    }

    [Fact]
    public void Impersonate()
    {
        var data = GetJwtAccessToken($"{Default.AdministratorEmailAddress}:{Default.CustomerEmailAddress}", Default.AdministratorPassword);
        Assert.True(data.statusCode == 200);
        Assert.True(data.accessToken.Length > 100);

        var expiresIn = (data.validUntil - DateTime.UtcNow).TotalSeconds;
        Assert.True(expiresIn > 500 && expiresIn < 600);
        Assert.True(Guid.TryParse(data.refreshToken, out Guid dummy));

        var authentications = Database.GetLastAuthentications(2);
        Assert.True(authentications.Last().UniqueID == Guid.Parse(Default.AdministratorAccountUniqueID));
        Assert.True(authentications.Last().IsSuccessful);
        Assert.True(!authentications.Last().IsShadowed);
        Assert.True(authentications.First().UniqueID == Guid.Parse(Default.CustomerAccountUniqueID));
        Assert.True(authentications.First().IsSuccessful);
        Assert.True(authentications.First().IsShadowed);
    }

    [Fact]
    public void ImpersonateNotAdmin()
    {
        var data = GetJwtAccessToken($"{Default.CustomerEmailAddress}:{Default.CustomerEmailAddress}", Default.AdministratorPassword);
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountCredentialsInvalid");

        var authentications = Database.GetLastAuthentications(1);
        Assert.True(authentications.First().UniqueID == Guid.Parse(Default.CustomerAccountUniqueID));
        Assert.True(!authentications.First().IsSuccessful);
        Assert.True(!authentications.First().IsShadowed);
    }

    [Fact]
    public void ImpersonateInvalidPassword()
    {
        var data = GetJwtAccessToken($"{Default.AdministratorEmailAddress}:{Default.CustomerEmailAddress}", "xxx");
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountCredentialsInvalid");

        var authentications = Database.GetLastAuthentications(1);
        Assert.True(authentications.First().UniqueID == Guid.Parse(Default.AdministratorAccountUniqueID));
        Assert.True(!authentications.First().IsSuccessful);
        Assert.True(!authentications.First().IsShadowed);
    }

    [Fact]
    public void ImpersonateInvalidRole()
    {
        var data = GetJwtAccessToken($"{Default.AdministratorEmailAddress}:{Default.AdministratorEmailAddress}", Default.AdministratorPassword);
        Assert.True(data.statusCode == 403);
        Assert.True(data.errorCodeName == "AccountRoleInvalidForOperation");

        var authentications = Database.GetLastAuthentications(2);
        Assert.True(authentications.Last().UniqueID == Guid.Parse(Default.AdministratorAccountUniqueID));
        Assert.True(authentications.Last().IsSuccessful);
        Assert.True(!authentications.Last().IsShadowed);
        Assert.True(authentications.First().UniqueID == Guid.Parse(Default.AdministratorAccountUniqueID));
        Assert.True(!authentications.First().IsSuccessful);
        Assert.True(authentications.First().IsShadowed);
    }

    [Fact]
    public void ImpersonateInvalidStatus()
    {
        Database.SetAccountPasswordExpiredTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        
        var data = GetJwtAccessToken($"{Default.AdministratorEmailAddress}:{Default.CustomerEmailAddress}", Default.AdministratorPassword);
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountCredentialsExpired");

        var authentications = Database.GetLastAuthentications(2);
        Assert.True(authentications.Last().UniqueID == Guid.Parse(Default.AdministratorAccountUniqueID));
        Assert.True(authentications.Last().IsSuccessful);
        Assert.True(!authentications.Last().IsShadowed);
        Assert.True(authentications.First().UniqueID == Guid.Parse(Default.CustomerAccountUniqueID));
        Assert.True(authentications.First().IsSuccessful);
        Assert.True(authentications.First().IsShadowed);
    }

    [Fact]
    public void ImpersonateLockedOut()
    {
        Database.SetAccountLockedOutTimestamp(Default.CustomerEmailAddress, DateTime.UtcNow);
        
        var data = GetJwtAccessToken($"{Default.AdministratorEmailAddress}:{Default.CustomerEmailAddress}", Default.AdministratorPassword);
        Assert.True(data.statusCode == 401);
        Assert.True(data.errorCodeName == "AccountLockedOutOverride");

        var authentications = Database.GetLastAuthentications(2);
        Assert.True(authentications.Last().UniqueID == Guid.Parse(Default.AdministratorAccountUniqueID));
        Assert.True(authentications.Last().IsSuccessful);
        Assert.True(!authentications.Last().IsShadowed);
        Assert.True(authentications.First().UniqueID == Guid.Parse(Default.CustomerAccountUniqueID));
        Assert.True(!authentications.First().IsSuccessful);
        Assert.True(authentications.First().IsShadowed);
    }

    [Fact]
    public void AccessTokenAttributes()
    {
        var data = GetJwtAccessToken();
        Assert.Equal(200, data.statusCode);
        Assert.True(data.role == "Customer");
        Assert.True(data.emailAddress == Default.CustomerEmailAddress); 
        Assert.True((data.validUntil - DateTime.UtcNow).TotalSeconds < 600);
        Assert.True(data.refreshToken != null);
        Assert.True(data.accessToken != null);  
        Assert.True(data.avatarUrl == "/content/images/avatars/03F83779-C888-4FEC-8B23-4DF160E52DFE.jpg");
        Assert.True(data.niceName == "Rex McCrie");
    }

    [Fact]
    public void UpdateNiceName()
    {
        var auth = GetJwtAccessToken(Default.AdministratorEmailAddress, Default.AdministratorPassword);
        var customerAccount = RequestWithJwtAccessToken($"administrator/accounts?emailAddress={Default.CustomerEmailAddress}", Method.Get, null, auth.accessToken);

        Assert.True(customerAccount.niceName == "Rex McCrie");

        var parameters = new
        {            
            name = "Harpeth Consulting",
            code = "HARPETH",
            firstName = "Peter",
            lastName = "Balliol",
            address = "1520 Horton Ave",
            address2 =  "Suite 403",
            city = "Nashville",
            state = "TN",
            postalCode = "60622",
            emailAddress = "customer@test.com",
            telephone = "+1 615-284-1128"
        };

        var update = RequestWithJwtAccessToken($"administrator/customers/{Default.CustomerUniqueID}", Method.Put, parameters, auth.accessToken);
        var updatedCustomerAccount = RequestWithJwtAccessToken($"administrator/accounts?emailAddress={Default.CustomerEmailAddress}", Method.Get, null, auth.accessToken);

        Assert.True(updatedCustomerAccount.niceName == "Peter Balliol");
    }
}

