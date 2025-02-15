CREATE TABLE [dbo].[AccountPassword] (
   [AccountPasswordID] [int] NOT NULL
      IDENTITY (1,1),
   [AccountID] [int] NOT NULL,
   [PasswordHash] [varchar](128) NOT NULL,
   [PasswordSalt] [varchar](8) NOT NULL,
   [ArchivedTimestamp] [datetime] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_AccountPassword] PRIMARY KEY CLUSTERED ([AccountPasswordID])
)

CREATE NONCLUSTERED INDEX [IX_AccountPassword] ON [dbo].[AccountPassword] ([AccountID])

GO
