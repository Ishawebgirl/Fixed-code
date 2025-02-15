ALTER TABLE [dbo].[ChatMessage] WITH CHECK ADD CONSTRAINT [FK_ChatMessage_Chat]
   FOREIGN KEY([ChatID]) REFERENCES [dbo].[Chat] ([ChatID])

GO
ALTER TABLE [dbo].[ChatMessage] WITH CHECK ADD CONSTRAINT [FK_ChatMessage_ChatRole]
   FOREIGN KEY([ChatRoleID]) REFERENCES [dbo].[ChatRole] ([ChatRoleID])

GO
