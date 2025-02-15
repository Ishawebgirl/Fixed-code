CREATE TABLE [dbo].[AccountAuthentication] (
   [AccountAuthenticationID] [int] NOT NULL
      IDENTITY (1,1),
   [AccountID] [int] NULL,
   [IdentityProviderID] [int] NOT NULL,
   [IsSuccessful] [bit] NOT NULL,
   [IsRefresh] [bit] NOT NULL,
   [IsShadowed] [bit] NOT NULL,
   [IpAddress] [varchar](50) NOT NULL,
   [AuthenticatedTimestamp] [datetime] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_AccountAuthentication] PRIMARY KEY CLUSTERED ([AccountAuthenticationID])
)

CREATE NONCLUSTERED INDEX [IX_AccountAuthentication] ON [dbo].[AccountAuthentication] ([AccountID])
CREATE NONCLUSTERED INDEX [IX_AccountAuthentication_1] ON [dbo].[AccountAuthentication] ([AuthenticatedTimestamp])

GO
