ALTER TABLE [dbo].[Claim] ADD CONSTRAINT [DF_Claim_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Claim] ADD CONSTRAINT [DF_Claim_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Claim] ADD CONSTRAINT [DF_Claim_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
