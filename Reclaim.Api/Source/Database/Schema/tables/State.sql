CREATE TABLE [dbo].[State] (
   [StateID] [int] NOT NULL,
   [Code] [nvarchar](10) NOT NULL,
   [Name] [nvarchar](200) NOT NULL,
   [Sequence] [int] NOT NULL,
   [IsEnabled] [bit] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_State] PRIMARY KEY CLUSTERED ([StateID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_State] ON [dbo].[State] ([Name])

GO
