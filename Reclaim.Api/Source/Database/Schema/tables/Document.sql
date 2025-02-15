CREATE TABLE [dbo].[Document] (
   [DocumentID] [int] NOT NULL
      IDENTITY (1,1),
   [DocumentTypeID] [int] NOT NULL,
   [ClaimID] [int] NULL,
   [AccountID] [int] NOT NULL,
   [Name] [nvarchar](400) NOT NULL,
   [Path] [nvarchar](450) NOT NULL,
   [Size] [int] NOT NULL,
   [Description] [nvarchar](max) NOT NULL,
   [Summary] [nvarchar](max) NULL,
   [Hash] [nvarchar](50) NOT NULL,
   [OriginatedTimestamp] [datetime] NULL,
   [UploadedTimestamp] [datetime] NOT NULL,
   [IngestedTimestamp] [datetime] NULL,
   [SummarizedTimestamp] [datetime] NULL,
   [TombstonedTimestamp] [datetime] NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [IX_Document_4] UNIQUE NONCLUSTERED ([Path])
   ,CONSTRAINT [PK_Document] PRIMARY KEY CLUSTERED ([DocumentID])
)

CREATE NONCLUSTERED INDEX [IX_Document] ON [dbo].[Document] ([Hash])
CREATE UNIQUE NONCLUSTERED INDEX [IX_Document_1] ON [dbo].[Document] ([UniqueID])
CREATE NONCLUSTERED INDEX [IX_Document_2] ON [dbo].[Document] ([ClaimID])
CREATE NONCLUSTERED INDEX [IX_Document_3] ON [dbo].[Document] ([AccountID])
CREATE NONCLUSTERED INDEX [IX_Document_5] ON [dbo].[Document] ([ClaimID], [TombstonedTimestamp])

GO
