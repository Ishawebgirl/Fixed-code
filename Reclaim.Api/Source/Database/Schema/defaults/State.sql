ALTER TABLE [dbo].[State] ADD CONSTRAINT [DF_State_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[State] ADD CONSTRAINT [DF_State_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
