ALTER TABLE [dbo].[ChatMessageCitation] WITH CHECK ADD CONSTRAINT [FK_ChatMessageCitation_ChatMessage]
   FOREIGN KEY([ChatMessageID]) REFERENCES [dbo].[ChatMessage] ([ChatMessageID])

GO
ALTER TABLE [dbo].[ChatMessageCitation] WITH CHECK ADD CONSTRAINT [FK_ChatMessageCitation_Document]
   FOREIGN KEY([DocumentID]) REFERENCES [dbo].[Document] ([DocumentID])

GO
