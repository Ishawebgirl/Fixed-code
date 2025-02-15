CREATE TABLE [dbo].[ChatMessage] (
   [ChatMessageID] [int] NOT NULL
      IDENTITY (1,1),
   [ChatID] [int] NOT NULL,
   [ChatRoleID] [int] NOT NULL,
   [Text] [nvarchar](max) NOT NULL,
   [SubmittedTimestamp] [datetime] NOT NULL,
   [ReceivedTimestamp] [datetime] NULL,
   [Metadata] [nvarchar](max) NULL,
   [IsError] [bit] NOT NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_ChatMessage] PRIMARY KEY CLUSTERED ([ChatMessageID])
)

CREATE NONCLUSTERED INDEX [IX_ChatMessage] ON [dbo].[ChatMessage] ([ChatID])

GO
