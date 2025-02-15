CREATE TABLE [dbo].[Role] (
   [RoleID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Role] PRIMARY KEY CLUSTERED ([RoleID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_Role] ON [dbo].[Role] ([Code])

GO
