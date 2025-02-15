CREATE TABLE [dbo].[ChatMessageCitation] (
   [ChatMessageCitationID] [int] NOT NULL
      IDENTITY (1,1),
   [ChatMessageID] [int] NOT NULL,
   [DocumentID] [int] NOT NULL,
   [PageNumber] [int] NOT NULL,
   [BoundingBoxes] [nvarchar](max) NOT NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_ChatMessageCitation] PRIMARY KEY CLUSTERED ([ChatMessageCitationID])
)


GO
