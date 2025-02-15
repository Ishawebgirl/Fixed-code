CREATE TABLE [dbo].[Chat] (
   [ChatID] [int] NOT NULL
      IDENTITY (1,1),
   [ChatTypeID] [int] NOT NULL,
   [AccountID] [int] NOT NULL,
   [ClaimID] [int] NOT NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [StartedTimestamp] [datetime] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Chat] PRIMARY KEY CLUSTERED ([ChatID])
)

CREATE NONCLUSTERED INDEX [IX_Chat] ON [dbo].[Chat] ([AccountID])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Chat_1] ON [dbo].[Chat] ([UniqueID])

GO
