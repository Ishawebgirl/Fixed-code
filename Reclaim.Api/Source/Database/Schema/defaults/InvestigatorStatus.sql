ALTER TABLE [dbo].[InvestigatorStatus] ADD CONSTRAINT [DF_InvestigatorStatus_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[InvestigatorStatus] ADD CONSTRAINT [DF_InvestigatorStatus_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
