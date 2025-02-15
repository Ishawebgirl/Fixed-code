using Reclaim.Api.Model;
using Azure.Storage.Blobs.Models;
using Azure.Storage.Blobs;
using Azure.AI.DocumentIntelligence;
using System.Security.Cryptography;
using Microsoft.EntityFrameworkCore;
using System.Text.RegularExpressions;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;
using Azure;

namespace Reclaim.Api.Services;

public class DocumentService : BaseService
{
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly SearchService _searchService;

    private enum ContainerName
    {
        Avatars,
        Documents,
    }

    private static Dictionary<string, DocumentType> _extensionMap = new Dictionary<string, DocumentType>
    {
        { ".docx", DocumentType.DOCX },
        { ".xlsx", DocumentType.XLSX },
        { ".jpg", DocumentType.JPG },
        { ".jpeg", DocumentType.JPG },
        { ".pdf", DocumentType.PDF },
        { ".mp4", DocumentType.MP4 }
    };

    public DocumentService(DatabaseContext db, IHttpContextAccessor httpContextAccessor, SearchService searchService)
    {
        _db = db;
        _httpContextAccessor = httpContextAccessor;
        _searchService = searchService;
    }

    private IQueryable<Document> GetQuery()
    {
        var (accountID, role) = GetCurrentAccountInfo(_httpContextAccessor);

        IQueryable<Document> query = _db.Documents
            .Include(x => x.Claim);

        switch (role)
        {
            case Role.Administrator:
                break;

            case Role.Customer:
                query = query.Where(x => x.Claim.Policy.Customer.AccountID == accountID);
                break;

            case Role.Investigator:
                query = query.Where(x => x.Claim.Investigator.AccountID == accountID);
                break;
        }

        return query;
    }

    private BlobContainerClient GetBlobContainerClient(ContainerName? containerName = ContainerName.Documents)
    {
        return new BlobContainerClient(
            Setting.AzureBlobStorageConnectionString,
            containerName?.ToString()?.ToLower() ?? "main");
    }

    private string GetShaHash(string fileName)
    {
        using (var fs = new FileStream(fileName, FileMode.Open))
        using (var bs = new BufferedStream(fs))
        {
            using (var sha1 = SHA1.Create())
            {
                var hash = sha1.ComputeHash(bs);
                var formatted = new StringBuilder(2 * hash.Length);

                foreach (byte b in hash)
                    formatted.AppendFormat("{0:X2}", b);

                return formatted.ToString();
            }
        }
    }

    private async Task<(string, string, DocumentType)> GetUniqueFileName(string fileName, Claim claim)
    {
        var existingFileNames = await _db.Documents
            .Where(x => x.Claim.Policy.Customer.ID == claim.Policy.Customer.ID)
            .Select(x => x.Name)
            .ToListAsync();

        var folder = Path.GetDirectoryName(fileName);
        var fileNameWithoutExtension = Path.GetFileNameWithoutExtension(fileName);
        var extension = Path.GetExtension(fileName).ToLower();

        if (!_extensionMap.ContainsKey(extension))
            throw new ApiException(ErrorCode.DocumentTypeNotSupported, $"Document type {extension} is not supported.");

        var documentType = _extensionMap[extension];
        extension = "." + documentType.ToString().ToLower();

        var number = 1;
        var regex = Regex.Match(fileName, @"(.+) \((\d+)\)\.\w+");

        if (regex.Success)
        {
            fileNameWithoutExtension = regex.Groups[1].Value;
            number = int.Parse(regex.Groups[2].Value);
        }

        while (existingFileNames.Contains(fileName))
        {
            number++;
            fileName = Path.Combine(folder, string.Format("{0} ({1}){2}", fileNameWithoutExtension, number, extension));
        }

        var path = $"/{claim.Policy.Customer.Code.ToUpper()}/{fileName}";

        return (fileName, path, documentType);
    }

    public async Task<Document> Get(Guid uniqueID)
    {
        var document = await GetQuery()
            .FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        if (document == null)
            throw new ApiException(ErrorCode.DocumentDoesNotExistForAccount, $"Document {uniqueID} does not exist or is not associated with the current account");

        return document;
    }

    public async Task<MemoryStream> Download(Document document)
    {
        var blobContainerClient = GetBlobContainerClient();
        var blobClient = blobContainerClient.GetBlobClient(document.Path);

        if (!blobClient.ExistsAsync().Result)
            throw new ApiException(ErrorCode.DocumentDownloadFromAzureFailed, $"Failed to find file {document.Name} at {document.Path} in blob storage.");

        var stream = new MemoryStream();
        using (var ms = new MemoryStream())
        {
            await blobClient.DownloadToAsync(ms);
            ms.Position = 0;
            ms.CopyTo(stream);
            stream.Position = 0;
        }

        return stream;
    }

    public async Task<List<string>> EnumerateBlobs(Customer customer)
    {
        var blobs = new List<string>();
        var blobContainerClient = GetBlobContainerClient();

        try
        {
            await foreach (var blobPage in blobContainerClient.GetBlobsAsync(BlobTraits.All, BlobStates.All, customer.Code.ToUpper()).AsPages())
            {
                blobs.AddRange(blobPage.Values.Select(x => x.Name));
            }
        }
        catch (RequestFailedException ex)
        {
            throw new ApiException(ErrorCode.DocumentEnumerationFromAzureFailed, $"Failed to enumerate files under {customer.Code} in blob storage. {ex.Message}");
        }

        return blobs;
    }

    public async Task<Document> Ingest(IFormFile formFile, DateTime originatedTimestamp, Claim claim)
    {
        var (accountID, role) = GetCurrentAccountInfo(_httpContextAccessor);
        var uniqueID = Guid.NewGuid();
        var fileName = formFile.FileName.Trim('\"');
        var tempFilePath = Path.GetTempFileName();

        try
        {
            using (var stream = new FileStream(tempFilePath, FileMode.Create))
            {
                await formFile.CopyToAsync(stream);
            }

            var hash = GetShaHash(tempFilePath);
            var hashMatch = await _db.Documents
                .Include(x => x.Claim.Policy.Customer)
                .Where(x => x.TombstonedTimestamp == null)
                .FirstOrDefaultAsync(x => x.Hash == hash);

            if (hashMatch != null)
                throw new ApiException(ErrorCode.DocumentHashAlreadyExists, $"Document with hash {hash} already exists for customer {hashMatch.Claim.Policy.Customer.Code}.");

            var (uniqueFileName, remoteFilePath, documentType) = await GetUniqueFileName(fileName, claim);

            await Upload(tempFilePath, remoteFilePath);
            
            // MAYDO add summarization

            var document = new Document
            {
                ClaimID = claim.ID,
                Type = documentType,
                AccountID = accountID,
                Name = uniqueFileName,
                Path = remoteFilePath,
                Size = Convert.ToInt32(new FileInfo(tempFilePath).Length),
                Description = uniqueFileName,
                Summary = null,
                Hash = hash,
                OriginatedTimestamp = originatedTimestamp,
                UploadedTimestamp = DateTime.UtcNow,
                UniqueID = uniqueID
            };

            _db.Documents.Add(document);
            await _db.SaveChangesAsync();

            await _searchService.AddEmbeddings(tempFilePath, remoteFilePath, claim, null as Investigator, document.ID, hash);

            document.IngestedTimestamp = DateTime.UtcNow;
            await _db.SaveChangesAsync();

            return document;
        }
        catch (Exception ex)
        {
            throw;
        }
        finally
        {
            File.Delete(tempFilePath);
        }
    }

    private async Task Upload(string localPath, string remotePath)
    {
        try
        {
            var blobContainerClient = GetBlobContainerClient();
            var blobClient = blobContainerClient.GetBlobClient(remotePath);

            var info = await blobClient.UploadAsync(localPath, overwrite: true);
        }
        catch (Exception ex)
        {
            throw new ApiException(ErrorCode.DocumentUploadToAzureFailed, ex.Message);
        }
    }

    /*
    public async Task<string> UploadAvatar(Account account, byte[] bytes, string fileName)
    {
        var tempFilePath = Path.GetTempFileName();
        File.WriteAllBytes(tempFilePath, bytes);

        var blobContainerClient = GetBlobContainerClient(ContainerName.Avatars);
        var blobClient = blobContainerClient.GetBlobClient(fileName);

        var info = await blobClient.UploadAsync(tempFilePath, overwrite: true);

        File.Delete(tempFilePath);
        account.AvatarUrl = blobClient.Uri.ToString();
        await _db.SaveChangesAsync();

        return account.AvatarUrl;
    }
    */

    public async Task<string> UploadAvatar(IFormFile formFile)
    {
        var (accountID, role) = GetCurrentAccountInfo(_httpContextAccessor);
        var extension = Path.GetExtension(formFile.FileName).ToLower();
        var fileName = $"{Guid.NewGuid().ToString().ToUpper()}.jpg";
        var tempFilePath = Path.GetTempFileName();

        try
        {
            var account = await _db.Accounts
                .FirstOrDefaultAsync(x => x.ID == accountID);

            using (var stream = new FileStream(tempFilePath, FileMode.Create))
            {
                await formFile.CopyToAsync(stream);
            }

            if (extension == ".png")
                ConvertPngToJpg(tempFilePath);

            ResizeAndCropJpg(tempFilePath, Setting.MaximumAvatarDimension, Setting.MaximumAvatarDimension);

            var blobContainerClient = GetBlobContainerClient(ContainerName.Avatars);
            var blobClient = blobContainerClient.GetBlobClient(fileName);

            var info = await blobClient.UploadAsync(tempFilePath, overwrite: true);

            account.AvatarUrl = blobClient.Uri.ToString();
            await _db.SaveChangesAsync();

            return account.AvatarUrl;
        }
        catch (Exception ex)
        {
            throw;
        }
        finally
        {
            File.Delete(tempFilePath);
        }
    }

    private void ConvertPngToJpg(string tempFilePath)
    {
        using (var image = Image.Load(tempFilePath))
        {
            image.Save(tempFilePath, new JpegEncoder());
        }
    }

    private void ResizeAndCropJpg(string tempFilePath, int maxWidth, int maxHeight)
    {
        using (var image = Image.Load(tempFilePath))
        {
            var aspectRatio = (double)image.Width / image.Height;
            var newSize = image.Width < image.Height
                ? new Size(maxWidth, (int)(maxWidth / aspectRatio))
                : new Size((int)(maxHeight * aspectRatio), maxHeight);

            image.Mutate(x => x.Resize(newSize));

            var cropRectangle = new Rectangle(
                (newSize.Width - maxWidth) / 2,
                (newSize.Height - maxHeight) / 2,
                Math.Min(maxWidth, newSize.Width),
                Math.Min(maxHeight, newSize.Height)
            );

            image.Mutate(x => x.Crop(cropRectangle));
            image.Save(tempFilePath, new JpegEncoder());
        }
    }

    public async Task Tombstone(Document document)
    {
        document.TombstonedTimestamp = DateTime.UtcNow;
        await _db.SaveChangesAsync();
    }

    internal async Task ConvertStreamToByteArray(MemoryStream stream)
    {
        throw new NotImplementedException();
    }
}