ALTER TABLE [dbo].[ClaimStatus] ADD CONSTRAINT [DF_ClaimStatus_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ClaimStatus] ADD CONSTRAINT [DF_ClaimStatus_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
