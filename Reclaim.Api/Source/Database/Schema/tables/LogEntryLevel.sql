CREATE TABLE [dbo].[LogEntryLevel] (
   [LogEntryLevelID] [int] NOT NULL,
   [Code] [nvarchar](50) NOT NULL,
   [Description] [nvarchar](200) NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_LogEntryLevel] PRIMARY KEY CLUSTERED ([LogEntryLevelID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_LogEntryLevel] ON [dbo].[LogEntryLevel] ([Code])

GO
