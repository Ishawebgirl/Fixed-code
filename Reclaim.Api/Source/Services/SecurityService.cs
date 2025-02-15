using Google.Apis.Auth;
using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;
using System.Data;
using System.Text;
using System.Text.RegularExpressions;

namespace Reclaim.Api.Services;

public class SecurityService : BaseService
{
    private readonly DocumentService _documentService;
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly EmailService _emailService;
    private readonly LogService _logService;

    public SecurityService(DocumentService documentService, DatabaseContext db, IHttpContextAccessor httpContextAccessor, LogService logService, EmailService emailService)
    {
        _documentService = documentService;
        _db = db;
        _httpContextAccessor = httpContextAccessor;
        _logService = logService;
    }

    public async Task<Account> GetAccount(string emailAddress)
    {
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.EmailAddress == emailAddress);

        return account;
    }

    public async Task<Account> GetAccount(Guid uniqueID)
    {
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        return account;
    }

    public async Task<Account> GetAccount(int id)
    {
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.ID == id);

        return account;
    }

    public async Task<Account> Authenticate(string emailAddress, string password, string ipAddress)
    {
        var error = $"Failed to authenticate as {emailAddress}";
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.EmailAddress == emailAddress);

        var authentication = new AccountAuthentication { Account = account, IdentityProvider = IdentityProvider.Local, IsSuccessful = false, IsRefresh = false, IsShadowed = false, IpAddress = ipAddress, AuthenticatedTimestamp = _db.ContextTimestamp };
        _db.AccountAuthentications.Add(authentication);

        if (account == null)
        {
            await _db.SaveChangesAsync();
            throw new ApiException(ErrorCode.AccountCredentialsInvalid, error);
        }

        _db.AccountAuthentications.Add(authentication);

        if (!CompareHashedPassword(account, password))
        {
            account.FailedAuthenticationCount++;

            if (account.FailedAuthenticationCount >= Setting.MaximumAuthenticationAttempts && account.LockedOutTimestamp == null)
                account.LockedOutTimestamp = _db.ContextTimestamp;

            await _db.SaveChangesAsync();
            throw new ApiException(ErrorCode.AccountCredentialsInvalid, error);
        }

        await _db.SaveChangesAsync();

        var apiException = await ValidateAccountStatus(account, IdentityProvider.Local, error);

        account.AuthenticatedTimestamp = _db.ContextTimestamp;
        account.SessionAuthenticatedTimestamp = _db.ContextTimestamp;
        account.LastActiveTimestamp = _db.ContextTimestamp;
        account.FailedAuthenticationCount = 0;
        authentication.IsSuccessful = true;
        await _db.SaveChangesAsync();

        if (apiException != null)
            throw apiException;

        return account;
    }

    public async Task<Account> AuthenticateAndImpersonate(string administratorEmailAddress, string administratorPassword, string ipAddress, string impersonatedEmailAddress)
    {
        var error = $"Failed to authenticate and impersonate {impersonatedEmailAddress} as {administratorEmailAddress}";
        var administratorAccount = await Authenticate(administratorEmailAddress, administratorPassword, ipAddress);

        if (administratorAccount.Role != Role.Administrator)
            throw new ApiException(ErrorCode.AccountCredentialsInvalid, $"Failed to authenticate as {administratorEmailAddress}.");

        var impersonatedAccount = await GetAccount(impersonatedEmailAddress);

        var authentication = new AccountAuthentication { Account = impersonatedAccount, IdentityProvider = IdentityProvider.Local, IsSuccessful = false, IsRefresh = false, IsShadowed = true, IpAddress = ipAddress, AuthenticatedTimestamp = _db.ContextTimestamp };
        _db.AccountAuthentications.Add(authentication);

        if (impersonatedAccount == null)
        {
            await _db.SaveChangesAsync();
            throw new ApiException(ErrorCode.AccountDoesNotExist, $"{error}, the account does not exist.");
        }

        if (!(new Role[] { Role.Customer, Role.Investigator, Role.Support }.Contains(impersonatedAccount.Role)))
        {
            await _db.SaveChangesAsync();
            throw new ApiException(ErrorCode.AccountRoleInvalidForOperation, $"{error}, the account role is {impersonatedAccount.Role}.");
        }

        var apiException = await ValidateAccountStatus(impersonatedAccount, IdentityProvider.Local, error);

        impersonatedAccount.AuthenticatedTimestamp = _db.ContextTimestamp;
        impersonatedAccount.SessionAuthenticatedTimestamp = _db.ContextTimestamp;
        impersonatedAccount.LastActiveTimestamp = _db.ContextTimestamp;
        impersonatedAccount.FailedAuthenticationCount = 0;
        authentication.IsSuccessful = true;
        await _db.SaveChangesAsync();

        if (apiException != null)
            throw apiException;

        return impersonatedAccount;
    }

    public async Task<Account> AuthenticateExternal(string emailAddress, IdentityProvider provider, string ipAddress)
    {
        var error = $"Failed to authenticate as {emailAddress}";
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.EmailAddress == emailAddress);

        var authentication = new AccountAuthentication { Account = account, IdentityProvider = provider, IsSuccessful = false, IsRefresh = false, IpAddress = ipAddress, AuthenticatedTimestamp = _db.ContextTimestamp };

        if (account == null)
            throw new ApiException(ErrorCode.AccountExternalCredentialsInvalid, error);

        _db.AccountAuthentications.Add(authentication);
        await _db.SaveChangesAsync();

        var apiException = await ValidateAccountStatus(account, provider, error);

        account.AuthenticatedTimestamp = _db.ContextTimestamp;
        account.SessionAuthenticatedTimestamp = _db.ContextTimestamp;
        account.LastActiveTimestamp = _db.ContextTimestamp;
        account.FailedAuthenticationCount = 0;
        authentication.IsSuccessful = true;
        await _db.SaveChangesAsync();

        if (apiException != null)
            throw apiException;

        return account;
    }

    private async Task<ApiException> ValidateAccountStatus(Account account, IdentityProvider provider, string error)
    {
        var ex = null as ApiException;

        switch (account.Status)
        {
            case AccountStatus.Tombstoned:
                await _db.SaveChangesAsync();
                throw new ApiException(ErrorCode.AccountTombstoned, $"{error}, the account is tombstoned.");

            case AccountStatus.LockedOut:
                await _db.SaveChangesAsync();
                if (account.FailedAuthenticationCount < Setting.MaximumAuthenticationAttempts)
                    throw new ApiException(ErrorCode.AccountLockedOutOverride, $"{error}, the account is locked out.");
                else
                    throw new ApiException(ErrorCode.AccountLockedOut, $"{error}, the account is locked out due to too many failed signins.");

            case AccountStatus.PasswordExpired:
                ex = new ApiException(ErrorCode.AccountCredentialsExpired, $"{error}, the account password is expired.");
                break;

            case AccountStatus.EmailAddressNotConfirmed:
                ex = new ApiException(ErrorCode.AccountEmailAddressNotConfirmed, $"{error}, the email address has not been confirmed.");
                break;

            case AccountStatus.Normal:

                if (account.LockedOutTimestamp != null && account.LockedOutTimestamp.Value.AddSeconds(Setting.AccountLockedOutTimeout) < _db.ContextTimestamp)
                {
                    await _logService.Add($"Removing account lockout on {account.EmailAddress} after {Setting.AccountLockedOutTimeout} seconds");
                    account.LockedOutTimestamp = null;
                    account.FailedAuthenticationCount = 0;
                }
                break;
        }

        if (account.IdentityProvider != provider)
        {
            switch (account.IdentityProvider)
            {
                case IdentityProvider.Local:
                    ex = new ApiException(ErrorCode.AccountRequiresIdentityProviderLocal, $"{error}, the account was created in the local database and cannot use the {provider} authorization workflow.");
                    break;

                case IdentityProvider.Google:
                    ex = new ApiException(ErrorCode.AccountRequiresIdentityProviderGoogle, $"{error}, the account was created via the Google authorization workflow and must use this provider to authenticate.");
                    break;
            }
        }

        return ex;
    }

    private static bool CompareHashedPassword(Account account, string password)
    {
        return CompareHashedPassword(account.PasswordHash, account.PasswordSalt, password);
    }

    private static bool CompareHashedPassword(string hash, string salt, string password)
    {
        var hashedString = new HashedString { Hash = hash, Salt = salt };

        return OneWayEncryption.Validate(hashedString, password);
    }

    public async Task RecordAccessTokenRefresh(int accountID, string ipAddress)
    {
        var account = _db.Accounts.First(x => x.ID == accountID);
        var authentication = new AccountAuthentication { AccountID = account.ID, IdentityProvider = account.IdentityProvider, IsSuccessful = true, IsRefresh = true, IpAddress = ipAddress, AuthenticatedTimestamp = _db.ContextTimestamp };

        account.SessionAuthenticatedTimestamp = _db.ContextTimestamp;
        account.LastActiveTimestamp = _db.ContextTimestamp;

        _db.AccountAuthentications.Add(authentication);
        await _db.SaveChangesAsync();
    }

    public async Task<Account> Confirm(string emailAddress, Guid token)
    {
        var error = $"Failed to confirm the account for {emailAddress}";
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.EmailAddress == emailAddress);

        if (account == null)
            throw new ApiException(ErrorCode.AccountCredentialsInvalid, error);

        if (account.MagicUrlToken == null)
            throw new ApiException(ErrorCode.AccountAlreadyConfirmed, $"{error}, the account is already confirmed.");

        if (account.MagicUrlToken.Value.CompareTo(token) != 0)
            throw new ApiException(ErrorCode.AccountMagicUrlTokenInvalid, $"{error}, the magic URL token provided in the request does not match the stored value.");

        if (account.MagicUrlValidUntil == null || _db.ContextTimestamp > account.MagicUrlValidUntil.Value)
            throw new ApiException(ErrorCode.AccountMagicUrlTokenExpired, $"{error}, the magic URL token has expired.");

        account.MagicUrlToken = null;
        account.MagicUrlValidUntil = null;
        account.EmailAddressConfirmedTimestamp = _db.ContextTimestamp;
        await _db.SaveChangesAsync();

        return account;
    }

    public async Task<Account> RequestPasswordReset(string emailAddress)
    {
        var error = $"Failed to request a password reset for {emailAddress}";
        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.EmailAddress == emailAddress);

        if (account == null)
            throw new ApiException(ErrorCode.AccountDoesNotExist, $"{error}, the account does not exist.");

        if (account.Role != Role.Customer)
            throw new ApiException(ErrorCode.AccountRoleInvalidForOperation, $"{error}, an account with role {account.Role} is not allowed to make this request.");

        if (account.IdentityProvider != IdentityProvider.Local)
        {
            await _emailService.SendRequestPasswordResetGoogle(account);
            throw new ApiException(ErrorCode.AccountRequiresIdentityProviderLocal, $"{error}, the account was created using the {account.IdentityProvider} workflow.");
        }

        switch (account.Status)
        {
            case AccountStatus.LockedOut:
            case AccountStatus.Tombstoned:
                // ?? case AccountStatus.EmailAddressNotConfirmed: 
                throw new ApiException(ErrorCode.AccountStatusInvalidForOperation, $"{error}, an account in {account.Status} status is not allowed to make this request.");
        }

        account.MagicUrlToken = Guid.NewGuid();
        account.MagicUrlValidUntil = _db.ContextTimestamp.AddSeconds(Setting.MagicUrlTimeout);
        account.PasswordExpiredTimestamp = _db.ContextTimestamp;
        await _db.SaveChangesAsync();

        var reloadedAccount = await GetAccount(account.ID);

        await _emailService.SendRequestPasswordReset(reloadedAccount);

        return reloadedAccount;
    }

    public async Task<Account> ResetPassword(string emailAddress, string newPassword, Guid token)
    {
        var error = $"Failed to reset the password for {emailAddress}";

        if (string.IsNullOrWhiteSpace(emailAddress) || string.IsNullOrEmpty(newPassword))
            throw new ApiException(ErrorCode.RequiredParameterNullOrEmpty, $"{error}, at least one required value was null or empty.");

        var account = await _db.Accounts.FirstOrDefaultAsync(x => x.EmailAddress == emailAddress);

        if (account == null)
            throw new ApiException(ErrorCode.AccountDoesNotExist, $"{error}, the account does not exist.");

        if (account.Role == Role.Administrator)
            throw new ApiException(ErrorCode.AccountRoleInvalidForOperation, $"{error}, an account with role {account.Role} is not allowed to make this request.");

        if (account.IdentityProvider != IdentityProvider.Local)
            throw new ApiException(ErrorCode.AccountRequiresIdentityProviderLocal, $"{error}, the account was created using the {account.IdentityProvider} workflow.");

        if (account.Status == AccountStatus.Tombstoned || account.Status == AccountStatus.LockedOut)
            throw new ApiException(ErrorCode.AccountStatusInvalidForOperation, $"{error}, an account with status {account.Status} is not allowed to make this request.");

        if (account.MagicUrlToken == null || account.MagicUrlToken.Value.CompareTo(token) != 0)
            throw new ApiException(ErrorCode.AccountMagicUrlTokenInvalid, $"{error}, the magic URL token provided in the request does not match the stored value.");

        if (account.MagicUrlValidUntil == null || _db.ContextTimestamp > account.MagicUrlValidUntil.Value)
            throw new ApiException(ErrorCode.AccountMagicUrlTokenExpired, $"{error}, the magic URL token has expired.");

        if (!Regex.Match(newPassword, RegularExpression.Password).Success)
            throw new ApiException(ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity, $"{error}, the password does not meet minimum complexity rules.");

        if (Setting.EnforcePasswordUniqueness)
        {
            if (CompareHashedPassword(account.PasswordHash, account.PasswordSalt, newPassword))
                throw new ApiException(ErrorCode.AccountPasswordUsedPreviously, $"{error}, the password has been used previously.");

            var previousPasswords = _db.AccountPasswords.Where(x => x.AccountID == account.ID).OrderByDescending(x => x.ArchivedTimestamp);

            foreach (var item in previousPasswords)
            {
                if (CompareHashedPassword(item.PasswordHash, item.PasswordSalt, newPassword))
                    throw new ApiException(ErrorCode.AccountPasswordUsedPreviously, $"{error}, the password has been used previously.");
            }
        }

        var previousPassword = new AccountPassword { AccountID = account.ID, PasswordHash = account.PasswordHash, PasswordSalt = account.PasswordSalt, ArchivedTimestamp = _db.ContextTimestamp };
        _db.AccountPasswords.Add(previousPassword);

        var passwordAndSalt = OneWayEncryption.CreateHash(newPassword);

        account.PasswordHash = passwordAndSalt.Hash;
        account.PasswordSalt = passwordAndSalt.Salt;
        account.MagicUrlValidUntil = null;
        account.MagicUrlToken = null;
        account.EmailAddressConfirmedTimestamp = _db.ContextTimestamp;
        account.PasswordChangedTimestamp = _db.ContextTimestamp;
        account.PasswordExpiredTimestamp = null;
        await _db.SaveChangesAsync();

        var reloadedAccount = await GetAccount(account.ID);

        await _emailService.SendPasswordChangeConfirmation(reloadedAccount);

        return reloadedAccount;
    }

    public string GeneratePassword()
    {
        const string characters = "abcdefghjmnpqrstwxyz";
        const string numbers = "3456789";
        const string symbols = "$#@*!^";

        var lowerCaseSegment = ThreadSafeRandom.GetRandomCharacters(characters, 2);
        var upperCaseSegment = ThreadSafeRandom.GetRandomCharacters(characters.ToUpper(), 2);
        var numbersSegment = ThreadSafeRandom.GetRandomCharacters(numbers, 3);
        var symbolsSegment = ThreadSafeRandom.GetRandomCharacters(symbols, 1);

        var scrambled = string.Concat(lowerCaseSegment, upperCaseSegment, numbersSegment, symbolsSegment).Shuffle();

        return scrambled;
    }

    public Dtos.GoogleCredential ValidateGoogleCredential(string credential, string emailAddress, string? firstName = null, string? lastName = null)
    {
        var error = "Failed to interpret Google JWT credential";

        if (string.IsNullOrEmpty(credential))
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, no credential was supplied.");

        var payload = null as GoogleJsonWebSignature.Payload;
        var expiration = DateTime.UtcNow;

        try
        {
            payload = GoogleJsonWebSignature.ValidateAsync(credential, new GoogleJsonWebSignature.ValidationSettings()).Result;
        }
        catch (Exception ex)
        {
            var innerMessage = "";

            if (ex.Message != "One or more errors occurred." && ex.InnerException != null)
                innerMessage = ex.Message;
            else
                innerMessage = ex.InnerException.Message;

            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the credential is invalid. {innerMessage}");
        }

        if (payload == null)
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the payload was null.");

        if (payload.Email.CompareTo(emailAddress) != 0)
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the email address does not match.");

        if (firstName != null && payload.GivenName.CompareTo(firstName) != 0)
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the first name does not match.");

        if (lastName != null && payload.FamilyName.CompareTo(lastName) != 0)
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the last name does not match.");

        if (!payload.Audience.Equals(Setting.GoogleOAuthClientID))
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the client ID is invalid.");

        if (!payload.Issuer.EndsWith("accounts.google.com"))
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the issuer is not Google.");

        if (payload.ExpirationTimeSeconds == null)
            throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the expiration is null.");

        else
        {
            expiration = DateTimeOffset.FromUnixTimeSeconds((long)payload.ExpirationTimeSeconds).DateTime;

            if (DateTime.UtcNow > expiration)
                throw new ApiException(ErrorCode.GoogleJwtBearerTokenInvalid, $"{error}, the credential is expired.");
        }

        return new Dtos.GoogleCredential { EmailAddress = payload.Email, FirstName = payload.GivenName, LastName = payload.FamilyName, Expiration = expiration };
    }

    public async Task<List<Account>> GetAuthenticatedAccounts()
    {
        var start = DateTime.UtcNow.AddSeconds(-Setting.JwtAccessTokenTimeout);
        var accounts = await _db.Accounts.Where(x => x.LastActiveTimestamp > start).ToListAsync();

        return accounts;
    }

    public async Task<List<Administrator>> GetAdministrators()
    {
        var administrators = await _db.Administrators.ToListAsync();

        return administrators;
    }

    public void SetLastActiveTimestampSync(int accountID)
    {
        var account = _db.Accounts.FirstOrDefault(x => x.ID == accountID);

        if (account == null)
            return;

        account.LastActiveTimestamp = _db.ContextTimestamp;
        _db.SaveChanges();
    }

    public async Task<Account> Build(string emailAddress, string password, Role role, IdentityProvider provider, string niceName)
    {
        var error = "Failed to create account";
        var passwordExpiredTimestamp = _db.ContextTimestamp as DateTime?;
        var passwordChangedTimestamp = _db.ContextTimestamp as DateTime?;
        var isPasswordNull = string.IsNullOrWhiteSpace(password);

        await ValidateEmailAddress(emailAddress, error);

        if (string.IsNullOrWhiteSpace(password))
        {
            password = GeneratePassword();
            passwordChangedTimestamp = null;
        }
        else
            passwordExpiredTimestamp = null;

        if (!Regex.Match(password, RegularExpression.Password).Success)
            throw new ApiException(ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity, $"{error}, the password does not meet minimum complexity rules.");

        var passwordAndSalt = OneWayEncryption.CreateHash(password);

        var account = new Account { EmailAddress = emailAddress, PasswordHash = passwordAndSalt.Hash, PasswordSalt = passwordAndSalt.Salt, Role = role, IdentityProvider = provider, NiceName = niceName, PasswordExpiredTimestamp = passwordExpiredTimestamp, PasswordChangedTimestamp = passwordChangedTimestamp, MagicUrlToken = Guid.NewGuid(), MagicUrlValidUntil = DateTime.UtcNow.AddSeconds(Setting.MagicUrlTimeout) };

        return account;
    }

    public async Task ValidateEmailAddress(string requestedEmailAddress, string error)
    {
        var existingAccount = await GetAccount(requestedEmailAddress);

        if (existingAccount != null)
        {
            await _emailService.SendEmailAddressAlreadyInUse(existingAccount);
            throw new ApiException(ErrorCode.AccountEmailAddressAlreadyExists, $"{error}, the email address {requestedEmailAddress} is already in use.", "EmailAddress");
        }
    }

    public async Task ValidateEmailAddressUpdate(string requestedEmailAddress, Account account, string error)
    {
        if (string.Compare(requestedEmailAddress, account.EmailAddress, true) == 0)
            return;

        var existingAccount = await GetAccount(requestedEmailAddress);

        if (existingAccount == null)
            return;

        if (existingAccount.ID != account.ID)
            throw new ApiException(ErrorCode.AccountEmailAddressAlreadyExists, $"{error}, the email address {requestedEmailAddress} is already in use.", "EmailAddress");
    }
}
