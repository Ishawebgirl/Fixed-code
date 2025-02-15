CREATE TABLE [dbo].[JobEvent] (
   [JobEventID] [int] NOT NULL
      IDENTITY (1,1),
   [JobID] [int] NOT NULL,
   [LogEntryID] [int] NULL,
   [ItemCount] [int] NULL,
   [Message] [nvarchar](500) NULL,
   [StartedTimestamp] [datetime] NULL,
   [FinishedTimestamp] [datetime] NULL,
   [TimedOutTimestamp] [datetime] NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_JobEvent] PRIMARY KEY CLUSTERED ([JobEventID])
)

CREATE NONCLUSTERED INDEX [IX_JobEvent] ON [dbo].[JobEvent] ([JobID])
CREATE NONCLUSTERED INDEX [IX_JobEvent_1] ON [dbo].[JobEvent] ([LogEntryID])

GO
