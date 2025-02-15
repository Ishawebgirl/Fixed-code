CREATE TABLE [dbo].[Customer] (
   [CustomerID] [int] NOT NULL
      IDENTITY (1,1),
   [CustomerStatusID] [int] NOT NULL,
   [AccountID] [int] NOT NULL,
   [Code] [nvarchar](10) NOT NULL,
   [Name] [nvarchar](50) NOT NULL,
   [FirstName] [nvarchar](100) NOT NULL,
   [LastName] [nvarchar](100) NOT NULL,
   [Address] [nvarchar](100) NOT NULL,
   [Address2] [nvarchar](100) NULL,
   [City] [nvarchar](100) NOT NULL,
   [StateID] [int] NOT NULL,
   [PostalCode] [nvarchar](20) NOT NULL,
   [Telephone] [nvarchar](20) NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Customer] PRIMARY KEY CLUSTERED ([CustomerID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_Customer] ON [dbo].[Customer] ([Code])
CREATE NONCLUSTERED INDEX [IX_Customer_1] ON [dbo].[Customer] ([Name])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Customer_2] ON [dbo].[Customer] ([AccountID])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Customer_3] ON [dbo].[Customer] ([UniqueID])

GO
