CREATE TABLE [dbo].[Email] (
   [EmailID] [int] NOT NULL
      IDENTITY (1,1),
   [EmailStatusID] [int] NOT NULL,
   [EmailTemplateID] [int] NOT NULL,
   [AccountID] [int] NOT NULL,
   [DeliverAfter] [datetime] NULL,
   [DeliveredTimestamp] [datetime] NULL,
   [ReceivedTimestamp] [datetime] NULL,
   [FailedTimestamp] [datetime] NULL,
   [TombstonedTimestamp] [datetime] NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Email] PRIMARY KEY CLUSTERED ([EmailID])
)

CREATE NONCLUSTERED INDEX [IX_Email] ON [dbo].[Email] ([DeliverAfter])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Email_1] ON [dbo].[Email] ([UniqueID])
CREATE NONCLUSTERED INDEX [IX_Email_2] ON [dbo].[Email] ([AccountID])
CREATE NONCLUSTERED INDEX [IX_Email_3] ON [dbo].[Email] ([EmailStatusID], [DeliverAfter])

GO
