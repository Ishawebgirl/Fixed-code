CREATE TABLE [dbo].[LogEntry] (
   [LogEntryID] [int] NOT NULL
      IDENTITY (1,1),
   [LogEntryLevelID] [int] NOT NULL,
   [AccountID] [int] NULL,
   [ErrorCodeID] [int] NULL,
   [GeneratedTimestamp] [datetime] NOT NULL,
   [Url] [nvarchar](1000) NULL,
   [Text] [nvarchar](max) NOT NULL,
   [StackTrace] [nvarchar](max) NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_LogEntry] PRIMARY KEY CLUSTERED ([LogEntryID])
)

CREATE NONCLUSTERED INDEX [IX_LogEntry] ON [dbo].[LogEntry] ([AccountID])
CREATE NONCLUSTERED INDEX [IX_LogEntry_1] ON [dbo].[LogEntry] ([ErrorCodeID])

GO
