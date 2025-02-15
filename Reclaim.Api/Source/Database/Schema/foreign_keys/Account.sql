ALTER TABLE [dbo].[Account] WITH CHECK ADD CONSTRAINT [FK_Account_IdentityProvider]
   FOREIGN KEY([IdentityProviderID]) REFERENCES [dbo].[IdentityProvider] ([IdentityProviderID])

GO
ALTER TABLE [dbo].[Account] WITH CHECK ADD CONSTRAINT [FK_Account_Role]
   FOREIGN KEY([RoleID]) REFERENCES [dbo].[Role] ([RoleID])

GO
