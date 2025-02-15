using System;
using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

 
namespace Reclaim.Api.Model;

public partial class DatabaseContext : DbContext
{
    public DatabaseContext(DbContextOptions<DatabaseContext> options)
        : base(options)
    {
        this.ChangeTracker.LazyLoadingEnabled = false;
    }

    private readonly DateTime _contextTimestamp = DateTime.UtcNow;

    public DateTime ContextTimestamp { get { return _contextTimestamp;  } }
    public virtual DbSet<Account> Accounts { get; set; }

    public virtual DbSet<AccountAuthentication> AccountAuthentications { get; set; }

    public virtual DbSet<AccountPassword> AccountPasswords { get; set; }

    public virtual DbSet<Administrator> Administrators { get; set; }

    public virtual DbSet<ApplicationSetting> ApplicationSettings { get; set; }

    public virtual DbSet<Chat> Chats { get; set; }

    public virtual DbSet<ChatMessage> ChatMessages { get; set; }

    public virtual DbSet<ChatMessageCitation> ChatMessageCitations { get; set; }

    public virtual DbSet<Claim> Claims { get; set; }

    public virtual DbSet<Customer> Customers { get; set; }

    public virtual DbSet<Document> Documents { get; set; }

    public virtual DbSet<Email> Emails { get; set; }

    public virtual DbSet<EmailTemplate> EmailTemplates { get; set; }

    public virtual DbSet<Investigator> Investigators { get; set; }

    public virtual DbSet<Job> Jobs { get; set; }

    public virtual DbSet<JobEvent> JobEvents { get; set; }

    public virtual DbSet<LogEntry> LogEntries { get; set; }

    public virtual DbSet<Policy> Policies { get; set; }

    public virtual DbSet<State> States { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Account>(entity =>
        {
            // entity.HasKey(e => e.AccountId).HasAnnotation("SqlServer:FillFactor", 80);

            entity.ToTable("Account");

            entity.HasIndex(e => e.UniqueID, "IX_Account").IsUnique();

            entity.HasIndex(e => e.EmailAddress, "IX_Account_1").IsUnique();

            entity.HasIndex(e => e.AuthenticatedTimestamp, "IX_Account_2");

            entity.HasIndex(e => e.LastActiveTimestamp, "IX_Account_3");

            entity.HasIndex(e => e.SessionAuthenticatedTimestamp, "IX_Account_4");

            entity.Property(e => e.ID).HasColumnName("AccountID");
            entity.Property(e => e.AuthenticatedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.AvatarUrl).HasMaxLength(250);
            entity.Property(e => e.BouncedEmailTimestamp).HasColumnType("datetime");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.EmailAddress).HasMaxLength(250);
            entity.Property(e => e.EmailAddressConfirmedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.IdentityProvider).HasColumnName("IdentityProviderID");
            entity.Property(e => e.LastActiveTimestamp).HasColumnType("datetime");
            entity.Property(e => e.LockedOutTimestamp).HasColumnType("datetime");
            entity.Property(e => e.MagicUrlValidUntil).HasColumnType("datetime");
            entity.Property(e => e.NiceName)
                .HasMaxLength(200)
                .HasDefaultValue("X");
            entity.Property(e => e.PasswordChangedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.PasswordExpiredTimestamp).HasColumnType("datetime");
            entity.Property(e => e.PasswordHash).HasMaxLength(128);
            entity.Property(e => e.PasswordSalt).HasMaxLength(50);
            entity.Property(e => e.Role).HasColumnName("RoleID");
            entity.Property(e => e.SessionAuthenticatedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.TombstonedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
        });

        modelBuilder.Entity<AccountAuthentication>(entity =>
        {
            // entity.HasKey(e => e.AccountAuthenticationId).HasAnnotation("SqlServer:FillFactor", 80);

            entity.ToTable("AccountAuthentication");

            entity.HasIndex(e => e.AccountID, "IX_AccountAuthentication");

            entity.HasIndex(e => e.AuthenticatedTimestamp, "IX_AccountAuthentication_1");

            entity.Property(e => e.ID).HasColumnName("AccountAuthenticationID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.AuthenticatedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.IdentityProvider).HasColumnName("IdentityProviderID");
            entity.Property(e => e.IpAddress)
                .HasMaxLength(50)
                .IsUnicode(false);
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Authentications)
                .HasForeignKey(d => d.AccountID)
                .HasConstraintName("FK_AccountAuthentication_Account");
        });

        modelBuilder.Entity<AccountPassword>(entity =>
        {
            entity.ToTable("AccountPassword");

            entity.HasIndex(e => e.AccountID, "IX_AccountPassword");

            entity.Property(e => e.ID).HasColumnName("AccountPasswordID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.ArchivedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.PasswordHash)
                .HasMaxLength(128)
                .IsUnicode(false);
            entity.Property(e => e.PasswordSalt)
                .HasMaxLength(8)
                .IsUnicode(false);
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Passwords)
                .HasForeignKey(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_AccountPassword_Account");
        });

        modelBuilder.Entity<Administrator>(entity =>
        {
            entity.ToTable("Administrator");

            entity.Property(e => e.ID).HasColumnName("AdministratorID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.FirstName).HasMaxLength(100);
            entity.Property(e => e.LastName).HasMaxLength(100);
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Administrators)
                .HasForeignKey(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Administrator_Account");
        });

        modelBuilder.Entity<ApplicationSetting>(entity =>
        {
            entity.ToTable("ApplicationSetting");

            entity.HasIndex(e => e.Name, "IX_ApplicationSetting").IsUnique();

            entity.Property(e => e.ID)
                .ValueGeneratedNever()
                .HasColumnName("ApplicationSettingID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.Name).HasMaxLength(50);
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
        });

        modelBuilder.Entity<Chat>(entity =>
        {
            entity.ToTable("Chat");

            entity.HasIndex(e => e.AccountID, "IX_Chat");

            entity.HasIndex(e => e.UniqueID, "IX_Chat_1").IsUnique();

            entity.Property(e => e.ID).HasColumnName("ChatID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.Type).HasColumnName("ChatTypeID");
            entity.Property(e => e.ClaimID).HasColumnName("ClaimID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.StartedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Chats)
                .HasForeignKey(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Chat_Account");

            entity.HasOne(d => d.Claim).WithMany(p => p.Chats)
                .HasForeignKey(d => d.ClaimID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Chat_Claim");
        });

        modelBuilder.Entity<ChatMessage>(entity =>
        {
            entity.ToTable("ChatMessage");

            entity.HasIndex(e => e.ChatID, "IX_ChatMessage");

            entity.Property(e => e.ID).HasColumnName("ChatMessageID");
            entity.Property(e => e.ChatID).HasColumnName("ChatID");
            entity.Property(e => e.ChatRole).HasColumnName("ChatRoleID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.ReceivedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.SubmittedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Chat).WithMany(p => p.Messages)
                .HasForeignKey(d => d.ChatID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_ChatMessage_Chat");
        });

        modelBuilder.Entity<ChatMessageCitation>(entity =>
        {
            entity.ToTable("ChatMessageCitation");

            entity.Property(e => e.ID).HasColumnName("ChatMessageCitationID");
            entity.Property(e => e.ChatMessageID).HasColumnName("ChatMessageID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.DocumentID).HasColumnName("DocumentID");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.ChatMessage).WithMany(p => p.Citations)
                .HasForeignKey(d => d.ChatMessageID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_ChatMessageCitation_ChatMessage");

            entity.HasOne(d => d.Document).WithMany(p => p.ChatMessageCitations)
                .HasForeignKey(d => d.DocumentID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_ChatMessageCitation_Document");
        });

        modelBuilder.Entity<Claim>(entity =>
        {
            entity.ToTable("Claim");

            entity.HasIndex(e => e.UniqueID, "IX_Claim").IsUnique();

            entity.HasIndex(e => e.Status, "IX_Claim_1");

            entity.HasIndex(e => e.InvestigatorID, "IX_Claim_3");

            entity.Property(e => e.ID).HasColumnName("ClaimID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.AdjudicatedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.AmountAdjusted).HasColumnType("money");
            entity.Property(e => e.AmountPaid).HasColumnType("money");
            entity.Property(e => e.AmountRequested).HasColumnType("money");
            entity.Property(e => e.AmountSubmitted).HasColumnType("money");
            entity.Property(e => e.Disposition).HasColumnName("ClaimDispositionID");
            entity.Property(e => e.Status).HasColumnName("ClaimStatusID");
            entity.Property(e => e.Type).HasColumnName("ClaimTypeID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.ExternalID)
                .HasMaxLength(50)
                .HasColumnName("ExternalID");
            entity.Property(e => e.IngestedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.InvestigatorID).HasColumnName("InvestigatorID");
            entity.Property(e => e.PolicyID).HasColumnName("PolicyID");
            entity.Property(e => e.TombstonedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Claims)
                .HasForeignKey(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Claim_Account");

            entity.HasOne(d => d.Investigator).WithMany(p => p.Claims)
                .HasForeignKey(d => d.InvestigatorID)
                .HasConstraintName("FK_Claim_Investigator");

            entity.HasOne(d => d.Policy).WithMany(p => p.Claims)
                .HasForeignKey(d => d.PolicyID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Claim_Policy");
        });

        modelBuilder.Entity<Customer>(entity =>
        {
            entity.ToTable("Customer");

            entity.HasIndex(e => e.Code, "IX_Customer").IsUnique();

            entity.HasIndex(e => e.Name, "IX_Customer_1");

            entity.HasIndex(e => e.AccountID, "IX_Customer_2").IsUnique();

            entity.HasIndex(e => e.UniqueID, "IX_Customer_3").IsUnique();

            entity.Property(e => e.ID).HasColumnName("CustomerID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.Address).HasMaxLength(100);
            entity.Property(e => e.Address2).HasMaxLength(100);
            entity.Property(e => e.City).HasMaxLength(100);
            entity.Property(e => e.Code).HasMaxLength(10);
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.Status).HasColumnName("CustomerStatusID");
            entity.Property(e => e.FirstName).HasMaxLength(100);
            entity.Property(e => e.LastName).HasMaxLength(100);
            entity.Property(e => e.Name).HasMaxLength(50);
            entity.Property(e => e.PostalCode).HasMaxLength(20);
            entity.Property(e => e.StateID).HasColumnName("StateID");
            entity.Property(e => e.Telephone).HasMaxLength(20);
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithOne(p => p.Customer)
                .HasForeignKey<Customer>(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Customer_Account");

            entity.HasOne(d => d.State).WithMany(p => p.Customers)
                .HasForeignKey(d => d.StateID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Customer_State");
        });

        modelBuilder.Entity<Document>(entity =>
        {
            entity.ToTable("Document");

            entity.HasIndex(e => e.Hash, "IX_Document");

            entity.HasIndex(e => e.UniqueID, "IX_Document_1").IsUnique();

            entity.HasIndex(e => e.ClaimID, "IX_Document_2");

            entity.HasIndex(e => e.AccountID, "IX_Document_3");

            entity.HasIndex(e => e.Path, "IX_Document_4").IsUnique();

            entity.HasIndex(e => new { e.ClaimID, e.TombstonedTimestamp }, "IX_Document_5");

            entity.Property(e => e.ID).HasColumnName("DocumentID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.ClaimID).HasColumnName("ClaimID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.Type).HasColumnName("DocumentTypeID");
            entity.Property(e => e.Hash).HasMaxLength(50);
            entity.Property(e => e.IngestedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.Name).HasMaxLength(400);
            entity.Property(e => e.OriginatedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.SummarizedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.TombstonedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.UploadedTimestamp).HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Documents)
                .HasForeignKey(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Document_Account");

            entity.HasOne(d => d.Claim).WithMany(p => p.Documents)
                .HasForeignKey(d => d.ClaimID)
                .HasConstraintName("FK_Document_Claim");
        });

        modelBuilder.Entity<Email>(entity =>
        {
            entity.ToTable("Email");

            entity.HasIndex(e => e.DeliverAfter, "IX_Email");

            entity.HasIndex(e => e.UniqueID, "IX_Email_1").IsUnique();

            entity.HasIndex(e => e.AccountID, "IX_Email_2");

            entity.HasIndex(e => new { e.Status, e.DeliverAfter }, "IX_Email_3");

            entity.Property(e => e.ID).HasColumnName("EmailID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.DeliverAfter).HasColumnType("datetime");
            entity.Property(e => e.DeliveredTimestamp).HasColumnType("datetime");
            entity.Property(e => e.Status).HasColumnName("EmailStatusID");
            entity.Property(e => e.TemplateID).HasColumnName("EmailTemplateID");
            entity.Property(e => e.FailedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.ReceivedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.TombstonedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithMany(p => p.Emails)
                .HasForeignKey(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Email_Account");

            entity.HasOne(d => d.Template).WithMany(p => p.Emails)
                .HasForeignKey(d => d.TemplateID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Email_EmailTemplate");
        });

        modelBuilder.Entity<EmailTemplate>(entity =>
        {
            entity.ToTable("EmailTemplate");

            entity.HasIndex(e => e.Code, "IX_EmailTemplate").IsUnique();

            entity.Property(e => e.ID)
                .ValueGeneratedNever()
                .HasColumnName("EmailTemplateID");
            entity.Property(e => e.ActionButtonText).HasMaxLength(50);
            entity.Property(e => e.Code).HasMaxLength(50);
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.ExternalID)
                .HasMaxLength(100)
                .HasColumnName("ExternalID");
            entity.Property(e => e.HighlightColor).HasMaxLength(10);
            entity.Property(e => e.Preheader)
                .HasMaxLength(100)
                .HasDefaultValue("x");
            entity.Property(e => e.Subject)
                .HasMaxLength(100)
                .HasDefaultValue("d");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
        });

        modelBuilder.Entity<Investigator>(entity =>
        {
            entity.ToTable("Investigator");

            entity.HasIndex(e => e.AccountID, "IX_Investigator").IsUnique();

            entity.HasIndex(e => e.UniqueID, "IX_Investigator_1").IsUnique();

            entity.Property(e => e.ID).HasColumnName("InvestigatorID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.Address).HasMaxLength(100);
            entity.Property(e => e.Address2).HasMaxLength(100);
            entity.Property(e => e.City).HasMaxLength(100);
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.FirstName).HasMaxLength(100);
            entity.Property(e => e.Status).HasColumnName("InvestigatorStatusID");
            entity.Property(e => e.LastName).HasMaxLength(100);
            entity.Property(e => e.PostalCode).HasMaxLength(20);
            entity.Property(e => e.StateID).HasColumnName("StateID");
            entity.Property(e => e.Telephone).HasMaxLength(20);
            entity.Property(e => e.UniqueID).HasDefaultValueSql("(newid())");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Account).WithOne(p => p.Investigator)
                .HasForeignKey<Investigator>(d => d.AccountID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Investigator_Account");

            entity.HasOne(d => d.State).WithMany(p => p.Investigators)
                .HasForeignKey(d => d.StateID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Investigator_State");
        });

        modelBuilder.Entity<Job>(entity =>
        {
            entity.ToTable("Job");

            entity.Property(e => e.ID)
                .ValueGeneratedNever()
                .HasColumnName("JobID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.Description).HasMaxLength(500);
            entity.Property(e => e.Status).HasColumnName("JobStatusID");
            entity.Property(e => e.Type).HasColumnName("JobTypeID");
            entity.Property(e => e.Name).HasMaxLength(50);
            entity.Property(e => e.NextEvent).HasColumnType("datetime");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
        });

        modelBuilder.Entity<JobEvent>(entity =>
        {
            // entity.HasKey(e => e.JobEventId).HasAnnotation("SqlServer:FillFactor", 80);

            entity.ToTable("JobEvent");

            entity.HasIndex(e => e.JobID, "IX_JobEvent");

            entity.HasIndex(e => e.LogEntryID, "IX_JobEvent_1");

            entity.Property(e => e.ID).HasColumnName("JobEventID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.FinishedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.JobID).HasColumnName("JobID");
            entity.Property(e => e.LogEntryID).HasColumnName("LogEntryID");
            entity.Property(e => e.Message).HasMaxLength(500);
            entity.Property(e => e.StartedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.TimedOutTimestamp).HasColumnType("datetime");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Job).WithMany(p => p.Events)
                .HasForeignKey(d => d.JobID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_JobEvent_Job");

            entity.HasOne(d => d.LogEntry).WithMany(p => p.JobEvents)
                .HasForeignKey(d => d.LogEntryID)
                .HasConstraintName("FK_JobEvent_LogEntry");
        });

        modelBuilder.Entity<LogEntry>(entity =>
        {
            // entity.HasKey(e => e.LogEntryId).HasAnnotation("SqlServer:FillFactor", 80);

            entity.ToTable("LogEntry");

            entity.HasIndex(e => e.AccountID, "IX_LogEntry");

            entity.HasIndex(e => e.ErrorCode, "IX_LogEntry_1");

            entity.Property(e => e.ID).HasColumnName("LogEntryID");
            entity.Property(e => e.AccountID).HasColumnName("AccountID");
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.ErrorCode).HasColumnName("ErrorCodeID");
            entity.Property(e => e.GeneratedTimestamp).HasColumnType("datetime");
            entity.Property(e => e.Level).HasColumnName("LogEntryLevelID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.Url).HasMaxLength(1000);

            entity.HasOne(d => d.Account).WithMany(p => p.LogEntries)
                .HasForeignKey(d => d.AccountID)
                .HasConstraintName("FK_LogEntry_Account");
        });

        modelBuilder.Entity<Policy>(entity =>
        {
            entity.ToTable("Policy");

            entity.HasIndex(e => e.CustomerID, "IX_Policy");

            entity.HasIndex(e => new { e.CustomerID, e.ExternalID }, "IX_Policy_1").IsUnique();

            entity.HasIndex(e => e.UniqueID, "IX_Policy_2").IsUnique();

            entity.HasIndex(e => e.StateID, "IX_Policy_3");

            entity.Property(e => e.ID).HasColumnName("PolicyID");
            entity.Property(e => e.Address).HasMaxLength(100);
            entity.Property(e => e.Address2).HasMaxLength(100);
            entity.Property(e => e.AnnualPremium).HasColumnType("money");
            entity.Property(e => e.City).HasMaxLength(100);
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.CustomerID).HasColumnName("CustomerID");
            entity.Property(e => e.Deductible).HasColumnType("money");
            entity.Property(e => e.ExternalID)
                .HasMaxLength(50)
                .HasColumnName("ExternalID");
            entity.Property(e => e.FirstName).HasMaxLength(100);
            entity.Property(e => e.LastName).HasMaxLength(100);
            entity.Property(e => e.OwnershipType).HasColumnName("OwnershipTypeID");
            entity.Property(e => e.PostalCode).HasMaxLength(20);
            entity.Property(e => e.PropertyType).HasColumnName("PropertyTypeID");
            entity.Property(e => e.RoofType).HasColumnName("RoofTypeID");
            entity.Property(e => e.StateID).HasColumnName("StateID");
            entity.Property(e => e.Telephone).HasMaxLength(20);
            entity.Property(e => e.UniqueID)
                .HasDefaultValueSql("(newid())")
                .HasColumnName("UniqueID");
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");

            entity.HasOne(d => d.Customer).WithMany(p => p.Policies)
                .HasForeignKey(d => d.CustomerID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Policy_Customer");

            entity.HasOne(d => d.State).WithMany(p => p.Policies)
                .HasForeignKey(d => d.StateID)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Policy_State");
        });

        modelBuilder.Entity<State>(entity =>
        {
            // entity.HasKey(e => e.StateId).HasAnnotation("SqlServer:FillFactor", 80);

            entity.ToTable("State");

            entity.HasIndex(e => e.Name, "IX_State").IsUnique();

            entity.Property(e => e.ID)
                .ValueGeneratedNever()
                .HasColumnName("StateID");
            entity.Property(e => e.Code).HasMaxLength(10);
            entity.Property(e => e.CreatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
            entity.Property(e => e.Name).HasMaxLength(200);
            entity.Property(e => e.UpdatedTimestamp)
                .HasDefaultValueSql("(getutcdate())")
                .HasColumnType("datetime");
        });

        OnModelCreatingPartial(modelBuilder);
    }
    
    protected override void ConfigureConventions(ModelConfigurationBuilder builder)
    {
        builder.Properties<DateOnly>()
            .HaveConversion<DateOnlyConverter>()
            .HaveColumnType("date");
    }

    public override int SaveChanges()
    { 
        var entries = ChangeTracker
            .Entries()
            .Where(e => e.Entity is Base && e.State == EntityState.Modified);

        foreach (var entityEntry in entries)
            ((Base)entityEntry.Entity).UpdatedTimestamp = DateTime.UtcNow;

        return base.SaveChanges();
    }


    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}

public class DateOnlyConverter : ValueConverter<DateOnly, DateTime>
{
    public DateOnlyConverter() : base(
            d => d.ToDateTime(TimeOnly.MinValue),
            d => DateOnly.FromDateTime(d))
    { }
}

