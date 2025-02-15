CREATE TABLE [dbo].[EmailStatus] (
   [EmailStatusID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](500) NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_EmailStatus] PRIMARY KEY CLUSTERED ([EmailStatusID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_EmailStatus] ON [dbo].[EmailStatus] ([Code])

GO
