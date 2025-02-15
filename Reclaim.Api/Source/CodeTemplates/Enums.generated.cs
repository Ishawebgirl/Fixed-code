
using System;
using System.ComponentModel.DataAnnotations;
using Reclaim.Api.Model;

namespace Reclaim.Api.Model
{
	public enum ChatRole : int
    {
		 [EnumDisplayName(DisplayName = "System")]
		 System = 1,

		 [EnumDisplayName(DisplayName = "User")]
		 User = 2,

		 [EnumDisplayName(DisplayName = "Assistant")]
		 Assistant = 3  
	}

	public enum ChatType : int
    {
		 [EnumDisplayName(DisplayName = "Claim query")]
		 ClaimQuery = 1  
	}

	public enum ClaimDisposition : int
    {
		 [EnumDisplayName(DisplayName = "Undecided")]
		 Undecided = 1,

		 [EnumDisplayName(DisplayName = "Not fraudulent")]
		 NotFraudulent = 2,

		 [EnumDisplayName(DisplayName = "Fraudulent")]
		 Fraudulent = 3  
	}

	public enum ClaimStatus : int
    {
		 [EnumDisplayName(DisplayName = "Unassigned")]
		 Unassigned = 1,

		 [EnumDisplayName(DisplayName = "Investigating")]
		 Investigating = 2,

		 [EnumDisplayName(DisplayName = "Adjudicated")]
		 Adjudicated = 3,

		 [EnumDisplayName(DisplayName = "Resolved")]
		 Resolved = 4,

		 [EnumDisplayName(DisplayName = "Tombstoned")]
		 Tombstoned = 5  
	}

	public enum ClaimType : int
    {
		 [EnumDisplayName(DisplayName = "Water")]
		 Water = 1,

		 [EnumDisplayName(DisplayName = "Fire")]
		 Fire = 2,

		 [EnumDisplayName(DisplayName = "Storm")]
		 Storm = 3,

		 [EnumDisplayName(DisplayName = "Theft")]
		 Theft = 4,

		 [EnumDisplayName(DisplayName = "Vandalism")]
		 Vandalism = 5,

		 [EnumDisplayName(DisplayName = "Mold")]
		 Mold = 6,

		 [EnumDisplayName(DisplayName = "Hail")]
		 Hail = 7,

		 [EnumDisplayName(DisplayName = "Other")]
		 Other = 8  
	}

	public enum CustomerStatus : int
    {
		 [EnumDisplayName(DisplayName = "Active")]
		 Active = 1,

		 [EnumDisplayName(DisplayName = "Uncommitted")]
		 Uncommitted = 2,

		 [EnumDisplayName(DisplayName = "Terminated")]
		 Terminated = 3  
	}

	public enum DocumentType : int
    {
		 [EnumDisplayName(DisplayName = "PDF")]
		 PDF = 1,

		 [EnumDisplayName(DisplayName = "MP4")]
		 MP4 = 2,

		 [EnumDisplayName(DisplayName = "JPG")]
		 JPG = 3,

		 [EnumDisplayName(DisplayName = "PNG")]
		 PNG = 4,

		 [EnumDisplayName(DisplayName = "DOCX")]
		 DOCX = 5,

		 [EnumDisplayName(DisplayName = "XLSX")]
		 XLSX = 6  
	}

	public enum EmailStatus : int
    {
		 [EnumDisplayName(DisplayName = "Pending")]
		 Pending = 1,

		 [EnumDisplayName(DisplayName = "Delivered")]
		 Delivered = 2,

		 [EnumDisplayName(DisplayName = "Failed")]
		 Failed = 3,

		 [EnumDisplayName(DisplayName = "Tombstoned")]
		 Tombstoned = 4  
	}

	public enum ErrorCode : int
    {
		 [EnumDisplayName(DisplayName = "Unknown")]
		 Unknown = 1000,

		 [EnumDisplayName(DisplayName = "Unhandled")]
		 Unhandled = 1001,

		 [EnumDisplayName(DisplayName = "Null reference")]
		 NullReference = 1002,

		 [EnumDisplayName(DisplayName = "Access denied")]
		 AccessDenied = 1003,

		 [EnumDisplayName(DisplayName = "Not implemented")]
		 NotImplemented = 1004,

		 [EnumDisplayName(DisplayName = "Timed out")]
		 TimedOut = 1005,

		 [EnumDisplayName(DisplayName = "Entity does not exist")]
		 EntityDoesNotExist = 1006,

		 [EnumDisplayName(DisplayName = "Invalid request")]
		 InvalidRequest = 1007,

		 [EnumDisplayName(DisplayName = "File not found")]
		 FileNotFound = 1008,

		 [EnumDisplayName(DisplayName = "EntityFramework error")]
		 EntityFrameworkError = 1009,

		 [EnumDisplayName(DisplayName = "Request parameter not expected in query string")]
		 RequestParameterNotExpectedInQueryString = 1010,

		 [EnumDisplayName(DisplayName = "Required parameter null or empty")]
		 RequiredParameterNullOrEmpty = 1011,

		 [EnumDisplayName(DisplayName = "Required parameter could not be parsed")]
		 ParameterCouldNotBeParsed = 1012,

		 [EnumDisplayName(DisplayName = "Required parameter could not be parsed to enum")]
		 ParameterCouldNotBeParsedToEnum = 1013,

		 [EnumDisplayName(DisplayName = "Rate limit exceeded")]
		 RateLimitExceeded = 1014,

		 [EnumDisplayName(DisplayName = "Blacklisted")]
		 Blacklisted = 1015,

		 [EnumDisplayName(DisplayName = "Enum value invalid")]
		 EnumValueInvalid = 1016,

		 [EnumDisplayName(DisplayName = "Anonymous invocation not allowed")]
		 AnonymousInvocationNotAllowed = 1017,

		 [EnumDisplayName(DisplayName = "Model to dto mapping not supported")]
		 ModelToDtoMappingNotSupported = 1018,

		 [EnumDisplayName(DisplayName = "Method attribute missing")]
		 MethodAttributeMissing = 1019,

		 [EnumDisplayName(DisplayName = "Master data value does not exist")]
		 MasterDataValueDoesNotExist = 1020,

		 [EnumDisplayName(DisplayName = "Application settings invalid")]
		 ApplicationSettingsInvalid = 1021,

		 [EnumDisplayName(DisplayName = "Model validation failed ")]
		 ModelValidationFailed = 1022,

		 [EnumDisplayName(DisplayName = "Email delivery failed")]
		 EmailDeliveryFailed = 1100,

		 [EnumDisplayName(DisplayName = "Scheduled job time out")]
		 ScheduledJobTimeout = 1200,

		 [EnumDisplayName(DisplayName = "Account credentials invalid")]
		 AccountCredentialsInvalid = 2000,

		 [EnumDisplayName(DisplayName = "Account credentials invalid")]
		 AccountExternalCredentialsInvalid = 2001,

		 [EnumDisplayName(DisplayName = "Account does not exist")]
		 AccountDoesNotExist = 2002,

		 [EnumDisplayName(DisplayName = "Account locked out")]
		 AccountLockedOut = 2003,

		 [EnumDisplayName(DisplayName = "Account locked out override")]
		 AccountLockedOutOverride = 2004,

		 [EnumDisplayName(DisplayName = "Account tombstoned")]
		 AccountTombstoned = 2005,

		 [EnumDisplayName(DisplayName = "Account credentials expired")]
		 AccountCredentialsExpired = 2006,

		 [EnumDisplayName(DisplayName = "Account role invalid for operation")]
		 AccountRoleInvalidForOperation = 2007,

		 [EnumDisplayName(DisplayName = "Account credentials not confirmed")]
		 AccountCredentialsNotConfirmed = 2008,

		 [EnumDisplayName(DisplayName = "Account email address invalid")]
		 AccountEmailAddressInvalid = 2009,

		 [EnumDisplayName(DisplayName = "Account password does not meet minimum complexity")]
		 AccountPasswordDoesNotMeetMinimumComplexity = 2010,

		 [EnumDisplayName(DisplayName = "Account email address already exists")]
		 AccountEmailAddressAlreadyExists = 2011,

		 [EnumDisplayName(DisplayName = "Account email address or guid invalid")]
		 AccountEmailAddressOrGuidInvalid = 2012,

		 [EnumDisplayName(DisplayName = "Account email address not confirmed")]
		 AccountEmailAddressNotConfirmed = 2013,

		 [EnumDisplayName(DisplayName = "Account magic url token missing")]
		 AccountMagicUrlTokenMissing = 2014,

		 [EnumDisplayName(DisplayName = "Account magic url token invalid")]
		 AccountMagicUrlTokenInvalid = 2015,

		 [EnumDisplayName(DisplayName = "Account magic url token expired")]
		 AccountMagicUrlTokenExpired = 2016,

		 [EnumDisplayName(DisplayName = "Account status invalid for operation")]
		 AccountStatusInvalidForOperation = 2017,

		 [EnumDisplayName(DisplayName = "Account password used previously")]
		 AccountPasswordUsedPreviously = 2018,

		 [EnumDisplayName(DisplayName = "Account already confirmed")]
		 AccountAlreadyConfirmed = 2019,

		 [EnumDisplayName(DisplayName = "Account requires identity provider local")]
		 AccountRequiresIdentityProviderLocal = 2020,

		 [EnumDisplayName(DisplayName = "Account requires identity provider google")]
		 AccountRequiresIdentityProviderGoogle = 2021,

		 [EnumDisplayName(DisplayName = "Account requires identity provider apple")]
		 AccountRequiresIdentityProviderApple = 2022,

		 [EnumDisplayName(DisplayName = "Account requires identity provider linked in")]
		 AccountRequiresIdentityProviderLinkedIn = 2023,

		 [EnumDisplayName(DisplayName = "JWT unknown error")]
		 JwtUnknownError = 2100,

		 [EnumDisplayName(DisplayName = "JWT role invalid")]
		 JwtRoleInvalid = 2101,

		 [EnumDisplayName(DisplayName = "JWT bearer token invalid")]
		 JwtBearerTokenInvalid = 2102,

		 [EnumDisplayName(DisplayName = "JWT claim not present")]
		 JwtClaimNotPresent = 2103,

		 [EnumDisplayName(DisplayName = "JWT bearer token missing")]
		 JwtBearerTokenMissing = 2104,

		 [EnumDisplayName(DisplayName = "JWT Auth claim invalid")]
		 JwtClaimInvalid = 2105,

		 [EnumDisplayName(DisplayName = "JWT bearer token expired")]
		 JwtBearerTokenExpired = 2106,

		 [EnumDisplayName(DisplayName = "JWT refresh token invalid")]
		 JwtRefreshTokenInvalid = 2107,

		 [EnumDisplayName(DisplayName = "Google JWT bearer token invalid")]
		 GoogleJwtBearerTokenInvalid = 2200,

		 [EnumDisplayName(DisplayName = "Google JWT nonce invalid")]
		 GoogleJwtNonceInvalid = 2201,

		 [EnumDisplayName(DisplayName = "Customer does not exist")]
		 CustomerDoesNotExist = 3000,

		 [EnumDisplayName(DisplayName = "Customer internal ID could not be generated")]
		 CustomerInvalid = 3001,

		 [EnumDisplayName(DisplayName = "Customer address block incomplete")]
		 CustomerCodeAlreadyExists = 3002,

		 [EnumDisplayName(DisplayName = "Customer code generation failed")]
		 CustomerCodeGenerationFailed = 3003,

		 [EnumDisplayName(DisplayName = "CustomerClaimDoesNotExist")]
		 CustomerDoesNotExistForAccount = 3004,

		 [EnumDisplayName(DisplayName = "")]
		 InvestigatorDoesNotExist = 3100,

		 [EnumDisplayName(DisplayName = "Investigator does not exist")]
		 InvestigatorDoesNotExistForAccount = 3101,

		 [EnumDisplayName(DisplayName = "Investigator not associated to claim ")]
		 InvestigatorNotAssociatedToClaim = 3102,

		 [EnumDisplayName(DisplayName = "Document download from azure failed ")]
		 DocumentDownloadFromAzureFailed = 4000,

		 [EnumDisplayName(DisplayName = "Document enumeration from azure failed ")]
		 DocumentEnumerationFromAzureFailed = 4001,

		 [EnumDisplayName(DisplayName = "Document upload to azure failed ")]
		 DocumentUploadToAzureFailed = 4002,

		 [EnumDisplayName(DisplayName = "Document hash already exists ")]
		 DocumentHashAlreadyExists = 4003,

		 [EnumDisplayName(DisplayName = "Document type not supported ")]
		 DocumentTypeNotSupported = 4004,

		 [EnumDisplayName(DisplayName = "Document does not exist ")]
		 DocumentDoesNotExistForAccount = 4005,

		 [EnumDisplayName(DisplayName = "")]
		 DocumentEmbeddingsDoNotExistForClaim = 4006,

		 [EnumDisplayName(DisplayName = "")]
		 DocumentOpenAIQueryFailed = 4007,

		 [EnumDisplayName(DisplayName = "")]
		 DocumentNotAssociatedToClaim = 4008,

		 [EnumDisplayName(DisplayName = "")]
		 DocumentTextCouldNotBeExtracted = 4009,

		 [EnumDisplayName(DisplayName = "")]
		 DocumentOpenAIQueryNoResults = 4010,

		 [EnumDisplayName(DisplayName = "")]
		 ChatDoesNotExist = 4100,

		 [EnumDisplayName(DisplayName = "")]
		 ChatDoesNotExistForAccount = 4101,

		 [EnumDisplayName(DisplayName = "Claim does not exist ")]
		 ClaimDoesNotExist = 5000,

		 [EnumDisplayName(DisplayName = "")]
		 ClaimDoesNotExistForAccount = 5001  
	}

	public enum IdentityProvider : int
    {
		 [EnumDisplayName(DisplayName = "Local")]
		 Local = 1,

		 [EnumDisplayName(DisplayName = "Google")]
		 Google = 2  
	}

	public enum InvestigatorStatus : int
    {
		 [EnumDisplayName(DisplayName = "Active")]
		 Active = 1,

		 [EnumDisplayName(DisplayName = "On probation")]
		 OnProbation = 2,

		 [EnumDisplayName(DisplayName = "Terminated")]
		 Terminated = 3  
	}

	public enum JobStatus : int
    {
		 [EnumDisplayName(DisplayName = "Ready")]
		 Ready = 1,

		 [EnumDisplayName(DisplayName = "Running")]
		 Running = 2,

		 [EnumDisplayName(DisplayName = "Paused")]
		 Paused = 3,

		 [EnumDisplayName(DisplayName = "TimedOut")]
		 TimedOut = 4,

		 [EnumDisplayName(DisplayName = "Disabled")]
		 Disabled = 5  
	}

	public enum JobType : int
    {
		 [EnumDisplayName(DisplayName = "Run diagnostics")]
		 RunDiagnostics = 1,

		 [EnumDisplayName(DisplayName = "Deliver email")]
		 DeliverEmail = 2  
	}

	public enum LogEntryLevel : int
    {
		 [EnumDisplayName(DisplayName = "Default")]
		 Default = 1,

		 [EnumDisplayName(DisplayName = "Verbose")]
		 Verbose = 2,

		 [EnumDisplayName(DisplayName = "Trace")]
		 Trace = 3  
	}

	public enum OwnershipType : int
    {
		 [EnumDisplayName(DisplayName = "OwnerOccupied")]
		 OwnerOccupied = 1,

		 [EnumDisplayName(DisplayName = "Rented")]
		 Rented = 2,

		 [EnumDisplayName(DisplayName = "Investment")]
		 Investment = 3  
	}

	public enum PropertyType : int
    {
		 [EnumDisplayName(DisplayName = "House")]
		 House = 1,

		 [EnumDisplayName(DisplayName = "Condominium")]
		 Condominium = 2  
	}

	public enum Role : int
    {
		 [EnumDisplayName(DisplayName = "Administrator")]
		 Administrator = 1,

		 [EnumDisplayName(DisplayName = "Customer")]
		 Customer = 2,

		 [EnumDisplayName(DisplayName = "Investigator")]
		 Investigator = 3,

		 [EnumDisplayName(DisplayName = "Support")]
		 Support = 4  
	}

	public enum RoofType : int
    {
		 [EnumDisplayName(DisplayName = "Rolled")]
		 Rolled = 1,

		 [EnumDisplayName(DisplayName = "Built-up roofing")]
		 BUR = 2,

		 [EnumDisplayName(DisplayName = "Membrane")]
		 Membrane = 3,

		 [EnumDisplayName(DisplayName = "Asphalt shingles")]
		 AsphaltShingles = 4,

		 [EnumDisplayName(DisplayName = "Metal")]
		 Metal = 5,

		 [EnumDisplayName(DisplayName = "Shakes")]
		 Shakes = 6,

		 [EnumDisplayName(DisplayName = "Clay")]
		 Clay = 7,

		 [EnumDisplayName(DisplayName = "Concrete")]
		 Concrete = 8,

		 [EnumDisplayName(DisplayName = "Slate")]
		 Slate = 9,

		 [EnumDisplayName(DisplayName = "Other")]
		 Other = 10  
	}
}
