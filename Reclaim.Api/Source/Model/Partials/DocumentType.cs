namespace Reclaim.Api.Model;

public partial class Document
{
    public string ContentType
    {
        get
        {
            switch (this.Type)
            {
                case DocumentType.JPG:
                    return "image/jpeg";
                case DocumentType.PNG:
                    return "image/png";
                case DocumentType.PDF:
                    return "application/pdf";
                case DocumentType.MP4:
                    return "video/mp4";
                case DocumentType.XLSX:
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
                case DocumentType.DOCX:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
                default:
                    return "application/octet-stream";
            }
        }
    }
}