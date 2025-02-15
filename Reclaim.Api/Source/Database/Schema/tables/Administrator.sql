CREATE TABLE [dbo].[Administrator] (
   [AdministratorID] [int] NOT NULL
      IDENTITY (1,1),
   [AccountID] [int] NOT NULL,
   [FirstName] [nvarchar](100) NOT NULL,
   [LastName] [nvarchar](100) NOT NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Administrator] PRIMARY KEY CLUSTERED ([AdministratorID])
)


GO
