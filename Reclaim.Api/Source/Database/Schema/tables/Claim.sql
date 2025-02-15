CREATE TABLE [dbo].[Claim] (
   [ClaimID] [int] NOT NULL
      IDENTITY (1,1),
   [ClaimTypeID] [int] NOT NULL,
   [ClaimStatusID] [int] NOT NULL,
   [ClaimDispositionID] [int] NOT NULL,
   [AccountID] [int] NOT NULL,
   [PolicyID] [int] NOT NULL,
   [InvestigatorID] [int] NULL,
   [ExternalID] [nvarchar](50) NOT NULL,
   [AmountSubmitted] [money] NULL,
   [AmountAdjusted] [money] NULL,
   [AmountRequested] [money] NULL,
   [AmountPaid] [money] NULL,
   [EventDate] [date] NOT NULL,
   [EventTime] [time] NULL,
   [IngestedTimestamp] [datetime] NOT NULL,
   [AdjudicatedTimestamp] [datetime] NULL,
   [TombstonedTimestamp] [datetime] NULL,
   [UniqueID] [uniqueidentifier] NOT NULL,
   [CreatedTimestamp] [datetime] NOT NULL,
   [UpdatedTimestamp] [datetime] NOT NULL

   ,CONSTRAINT [PK_Claim] PRIMARY KEY CLUSTERED ([ClaimID])
)

CREATE UNIQUE NONCLUSTERED INDEX [IX_Claim] ON [dbo].[Claim] ([UniqueID])
CREATE NONCLUSTERED INDEX [IX_Claim_1] ON [dbo].[Claim] ([ClaimStatusID])
CREATE NONCLUSTERED INDEX [IX_Claim_3] ON [dbo].[Claim] ([InvestigatorID])

GO
