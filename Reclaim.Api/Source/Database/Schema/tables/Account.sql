CREATE TABLE [dbo].[Account] (
   [AccountID] [int] NOT NULL
      IDENTITY (1,1),
   [RoleID] [int] NOT NULL,
   [IdentityProviderID] [int] NOT NULL,
   [EmailAddress] [nvarchar](250) NOT NULL,
   [PasswordHash] [nvarchar](128) NOT NULL,
   [PasswordSalt] [nvarchar](50) NOT NULL,
   [AvatarUrl] [nvarchar](250) NULL,
   [NiceName] [nvarchar](200) NOT NULL,
   [MagicUrlToken] [uniqueidentifier] NULL,
   [MagicUrlValidUntil] [datetime] NULL,
   [AuthenticatedTimestamp] [datetime] NULL,
   [SessionAuthenticatedTimestamp] [datetime] NULL,
   [LastActiveTimestamp] [datetime] NULL,
   [FailedAuthenticationCount] [int] NOT NULL,
   [BouncedEmailCount] [int] NOT NULL,
   [BouncedEmailTimestamp] [datetime] NULL,
   [PasswordExpiredTimestamp] [datetime] NULL,
   [PasswordChangedTimestamp] [datetime] NULL,
   [EmailAddressConfirmedTimestamp] [datetime] NULL,
   [TombstonedTimestamp] [datetime] NULL,
   [LockedOutTimestamp] [datetime] NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Account] PRIMARY KEY CLUSTERED ([AccountID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_Account] ON [dbo].[Account] ([UniqueID])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Account_1] ON [dbo].[Account] ([EmailAddress])
CREATE NONCLUSTERED INDEX [IX_Account_2] ON [dbo].[Account] ([AuthenticatedTimestamp])
CREATE NONCLUSTERED INDEX [IX_Account_3] ON [dbo].[Account] ([LastActiveTimestamp])
CREATE NONCLUSTERED INDEX [IX_Account_4] ON [dbo].[Account] ([SessionAuthenticatedTimestamp])

GO
