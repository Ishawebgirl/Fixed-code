ALTER TABLE [dbo].[ClaimDisposition] ADD CONSTRAINT [DF_ClaimDisposition_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ClaimDisposition] ADD CONSTRAINT [DF_ClaimDisposition_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
