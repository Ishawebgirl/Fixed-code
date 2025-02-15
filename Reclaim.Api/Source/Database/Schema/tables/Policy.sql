CREATE TABLE [dbo].[Policy] (
   [PolicyID] [int] NOT NULL
      IDENTITY (1,1),
   [CustomerID] [int] NOT NULL,
   [ExternalID] [nvarchar](50) NOT NULL,
   [BindingDate] [date] NULL,
   [StartDate] [date] NULL,
   [EndDate] [date] NULL,
   [Deductible] [money] NULL,
   [AnnualPremium] [money] NULL,
   [ClaimsInLastYear] [int] NULL,
   [ClaimsInLast3Years] [int] NULL,
   [FirstName] [nvarchar](100) NOT NULL,
   [LastName] [nvarchar](100) NOT NULL,
   [Address] [nvarchar](100) NOT NULL,
   [Address2] [nvarchar](100) NULL,
   [City] [nvarchar](100) NOT NULL,
   [StateID] [int] NOT NULL,
   [PostalCode] [nvarchar](20) NOT NULL,
   [Telephone] [nvarchar](20) NOT NULL,
   [DateOfBirth] [date] NULL,
   [Bedrooms] [int] NOT NULL,
   [Bathrooms] [int] NULL,
   [OwnershipTypeID] [int] NOT NULL,
   [PropertyTypeID] [int] NOT NULL,
   [RoofTypeID] [int] NOT NULL,
   [YearBuilt] [int] NOT NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Policy] PRIMARY KEY CLUSTERED ([PolicyID])
)

CREATE NONCLUSTERED INDEX [IX_Policy] ON [dbo].[Policy] ([CustomerID])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Policy_1] ON [dbo].[Policy] ([CustomerID], [ExternalID])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Policy_2] ON [dbo].[Policy] ([UniqueID])
CREATE NONCLUSTERED INDEX [IX_Policy_3] ON [dbo].[Policy] ([StateID])

GO
