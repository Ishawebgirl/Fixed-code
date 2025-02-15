CREATE TABLE [dbo].[Investigator] (
   [InvestigatorID] [int] NOT NULL
      IDENTITY (1,1),
   [InvestigatorStatusID] [int] NOT NULL,
   [AccountID] [int] NOT NULL,
   [FirstName] [nvarchar](100) NOT NULL,
   [LastName] [nvarchar](100) NOT NULL,
   [Address] [nvarchar](100) NOT NULL,
   [Address2] [nvarchar](100) NULL,
   [City] [nvarchar](100) NOT NULL,
   [StateID] [int] NOT NULL,
   [PostalCode] [nvarchar](20) NOT NULL,
   [Telephone] [nvarchar](20) NOT NULL,
   [UniqueId] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Investigator] PRIMARY KEY CLUSTERED ([InvestigatorID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_Investigator] ON [dbo].[Investigator] ([AccountID])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Investigator_1] ON [dbo].[Investigator] ([UniqueId])

GO
