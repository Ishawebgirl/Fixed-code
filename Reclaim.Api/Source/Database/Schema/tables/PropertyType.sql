CREATE TABLE [dbo].[PropertyType] (
   [PropertyTypeID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_PropertyType] PRIMARY KEY CLUSTERED ([PropertyTypeID])
)


GO
