












using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Configuration;


namespace Reclaim.Api
{
    public enum SettingName
    {

		GoogleOAuthClientID,

		ScheduledJobPollingFrequency,

		CookieDomain,

		PhysicalSitePath,

		ApiRootUrl,

		WebsiteRootUrl,

		SaveStackTraceInLogEntry,

		DeliverUnhandledExceptionEmails,

		MaximumAuthenticationAttempts,

		SystemAdministratorAccountID,

		CopyrightNotice,

		LogEntryLevel,

		OneWayEncryptionPepper,

		SendGridApiUrl,

		SendGridApiKey,

		SendGridFromEmailAddress,

		SendGridFromName,

		MagicUrlTimeout,

		AccountLockedOutTimeout,

		EnforcePasswordUniqueness,

		EnforceRequestThrottling,

		JwtAccessTokenTimeout,

		MaximumEmailBatchCount,

		JwtAccessTokenSecret,

		SwaggerUiToken,

		MaximumAvatarDimension,

		AzureBlobStorageConnectionString,

		AzureOpenAIEndpoint,

		AzureOpenAIKey,

		AzureOpenAIEmbeddingDeploymentName,

		AzureOpenAIDeploymentName,

		AzureDocumentIntelligenceEndpoint,

		AzureDocumentIntelligenceKey,

		AzureCognitiveSearchEndpoint,

		AzureCognitiveSearchKey,

		AzureCognitiveSearchIndexName
	}

	public static class Setting
	{

		public static string GoogleOAuthClientID { get { return SettingManager.Get<string>(SettingName.GoogleOAuthClientID); } }

		public static int ScheduledJobPollingFrequency { get { return SettingManager.Get<int>(SettingName.ScheduledJobPollingFrequency); } }

		public static string CookieDomain { get { return SettingManager.Get<string>(SettingName.CookieDomain); } }

		public static string PhysicalSitePath { get { return SettingManager.Get<string>(SettingName.PhysicalSitePath); } }

		public static string ApiRootUrl { get { return SettingManager.Get<string>(SettingName.ApiRootUrl); } }

		public static string WebsiteRootUrl { get { return SettingManager.Get<string>(SettingName.WebsiteRootUrl); } }

		public static bool SaveStackTraceInLogEntry { get { return SettingManager.Get<bool>(SettingName.SaveStackTraceInLogEntry); } }

		public static bool DeliverUnhandledExceptionEmails { get { return SettingManager.Get<bool>(SettingName.DeliverUnhandledExceptionEmails); } }

		public static int MaximumAuthenticationAttempts { get { return SettingManager.Get<int>(SettingName.MaximumAuthenticationAttempts); } }

		public static int SystemAdministratorAccountID { get { return SettingManager.Get<int>(SettingName.SystemAdministratorAccountID); } }

		public static string CopyrightNotice { get { return SettingManager.Get<string>(SettingName.CopyrightNotice); } }

		public static int LogEntryLevel { get { return SettingManager.Get<int>(SettingName.LogEntryLevel); } }

		public static string OneWayEncryptionPepper { get { return SettingManager.Get<string>(SettingName.OneWayEncryptionPepper); } }

		public static string SendGridApiUrl { get { return SettingManager.Get<string>(SettingName.SendGridApiUrl); } }

		public static string SendGridApiKey { get { return SettingManager.Get<string>(SettingName.SendGridApiKey); } }

		public static string SendGridFromEmailAddress { get { return SettingManager.Get<string>(SettingName.SendGridFromEmailAddress); } }

		public static string SendGridFromName { get { return SettingManager.Get<string>(SettingName.SendGridFromName); } }

		public static int MagicUrlTimeout { get { return SettingManager.Get<int>(SettingName.MagicUrlTimeout); } }

		public static int AccountLockedOutTimeout { get { return SettingManager.Get<int>(SettingName.AccountLockedOutTimeout); } }

		public static bool EnforcePasswordUniqueness { get { return SettingManager.Get<bool>(SettingName.EnforcePasswordUniqueness); } }

		public static bool EnforceRequestThrottling { get { return SettingManager.Get<bool>(SettingName.EnforceRequestThrottling); } }

		public static int JwtAccessTokenTimeout { get { return SettingManager.Get<int>(SettingName.JwtAccessTokenTimeout); } }

		public static int MaximumEmailBatchCount { get { return SettingManager.Get<int>(SettingName.MaximumEmailBatchCount); } }

		public static string JwtAccessTokenSecret { get { return SettingManager.Get<string>(SettingName.JwtAccessTokenSecret); } }

		public static string SwaggerUiToken { get { return SettingManager.Get<string>(SettingName.SwaggerUiToken); } }

		public static int MaximumAvatarDimension { get { return SettingManager.Get<int>(SettingName.MaximumAvatarDimension); } }

		public static string AzureBlobStorageConnectionString { get { return SettingManager.Get<string>(SettingName.AzureBlobStorageConnectionString); } }

		public static string AzureOpenAIEndpoint { get { return SettingManager.Get<string>(SettingName.AzureOpenAIEndpoint); } }

		public static string AzureOpenAIKey { get { return SettingManager.Get<string>(SettingName.AzureOpenAIKey); } }

		public static string AzureOpenAIEmbeddingDeploymentName { get { return SettingManager.Get<string>(SettingName.AzureOpenAIEmbeddingDeploymentName); } }

		public static string AzureOpenAIDeploymentName { get { return SettingManager.Get<string>(SettingName.AzureOpenAIDeploymentName); } }

		public static string AzureDocumentIntelligenceEndpoint { get { return SettingManager.Get<string>(SettingName.AzureDocumentIntelligenceEndpoint); } }

		public static string AzureDocumentIntelligenceKey { get { return SettingManager.Get<string>(SettingName.AzureDocumentIntelligenceKey); } }

		public static string AzureCognitiveSearchEndpoint { get { return SettingManager.Get<string>(SettingName.AzureCognitiveSearchEndpoint); } }

		public static string AzureCognitiveSearchKey { get { return SettingManager.Get<string>(SettingName.AzureCognitiveSearchKey); } }

		public static string AzureCognitiveSearchIndexName { get { return SettingManager.Get<string>(SettingName.AzureCognitiveSearchIndexName); } }

	}
}

