import { useEffect, useMemo, useRef, useState } from 'react'
import * as Model from 'api/model'
import { useParams } from 'react-router-dom'
import Menu from 'components/Menu'
import MenuItem from 'components/MenuItem'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import FilterTableMenuItem from 'components/FilterTableMenuItem'
import Table from 'components/Table'
import { Log } from 'helpers/log'
import useConfirmation from 'hooks/useConfirmation'
import moment from 'moment'
import useNotification from 'hooks/useNotification'
import DocumentViewer from 'components/DocumentViewer'

const ClaimDocuments = () => {
  const columns = useMemo(
    () => [
      { label: '', accessor: 'type', type: 'documentType' },
      {
        label: 'Name / Type',
        accessor: 'name',
        type: 'document',
      },
      {
        label: 'Summary',
        accessor: 'summary',
      },
      {
        label: 'Size',
        accessor: 'size',
        type: 'fileSize',
      },
      {
        label: 'Originated',
        accessor: 'originatedTimestamp',
        type: 'datetime',
      },
      {
        label: 'Uploaded',
        accessor: 'uploadedTimestamp',
        type: 'datetime',
      },
    ],
    []
  )

  const claimApi = useMemo(() => new Model.ClaimClient(process.env.REACT_APP_API_URL), [])

  const tableRef = useRef<{ handleExport: () => void } | null>(null)
  const fileUploadRef = useRef<HTMLInputElement | null>(null)
  const { confirm } = useConfirmation()
  const { notify } = useNotification()
  const { uniqueID } = useParams()

  const [claim, setClaim] = useState<Model.Claim>()
  const [documents, setDocuments] = useState<Model.Document[]>()
  const [filterTerms, setFilterTerms] = useState('')
  const [clickedDocument, setClickedDocument] = useState<Model.Document>()

  const confirmDelete = (clickedDocument: Model.Document) => {
    if (!handleDelete) {
      return
    }

    confirm('Confirm deletion', 'Are you sure you want to delete this document?', () =>
      handleDelete(clickedDocument)
    )
  }

  const updateProgressBar = (percentComplete: number) => {
    const tr = document.getElementById('row-temp-id')
    if (tr) {
      tr.setAttribute(
        'style',
        `background-image: linear-gradient(to right, var(--color-highlight) ${percentComplete}%, transparent 0%);`
      )
    }
  }

  const handleFileChange = async (event: any) => {
    handleUpload(event.target.files[0], moment(event.target.files[0].lastModified))
    fileUploadRef.current!.value = ''

    for (let i = 0; i <= 10; i++) {
      updateProgressBar(i * 10)
      await new Promise((resolve) => setTimeout(resolve, 500))
    }
  }

  /*
  const handleRowClick = async (document: Model.Document) => {
    try {
      const response = await claimApi.downloadDocument(claim!.uniqueID, document.uniqueID)
      const a = window.document.createElement('a')
      a.href = window.URL.createObjectURL(response.data)
      a.download = document.name
      window.document.body.appendChild(a)
      a.click()
      a.remove()
    } catch (error: any) {
      Log.add(JSON.stringify(error))
    }
  }
  */

  const handleRowClick = async (document: Model.Document) => {
    setClickedDocument(document)
  }

  const handleDelete = async (document: Model.Document) => {
    try {
      await claimApi.tombstoneDocument(claim!.uniqueID, document.uniqueID)
      await reload()
    } catch (error: any) {
      Log.add(JSON.stringify(error))
    }
  }

  const handleUpload = async (file: File, timestamp: moment.Moment) => {
    if (file) {
      let fileParameter: Model.FileParameter = {
        data: file,
        fileName: file.name ?? 'Unknown',
      }

      try {
        const dummy = {
          uniqueID: 'temp-id',
          name: file.name,
          size: file.size,
          originatedTimestamp: timestamp.toISOString(),
          type: file.name.split('.').pop()?.toUpperCase() ?? 'unknown',
          path: '',
          hash: '',
          description: '',
          tombstonedTimestamp: undefined,
          createdTimestamp: timestamp.toISOString(),
          modifiedTimestamp: timestamp.toISOString(),
        } as unknown as Model.Document

        setDocuments([...documents!, dummy])
        await claimApi.uploadDocument(claim!.uniqueID, timestamp, fileParameter)
      } catch (error: any) {
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
          return
        }
        const apiError = JSON.parse(error.response)

        switch (apiError?.errorCodeName) {
          case Model.ErrorCode.DocumentHashAlreadyExists:
            notify('Error', 'A file with that hash already exists')
            break
        }
      } finally {
        await reload()
      }
    }
  }

  const reload = async () => {
    if (uniqueID === undefined) {
      return
    }

    await claimApi.getClaim(uniqueID).then((result) => {
      setClaim(result)
      setDocuments(result.documents.filter((x) => x.tombstonedTimestamp === undefined))
    })
  }

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uniqueID])

  return (
    claim &&
    documents && (
      <>
        <div className="header">Documents for claim {claim.externalID}</div>
        <Menu>
          <MenuItem label="Return to claim" icon="ArrowLeft" linkTo=".." />
          <MenuItem
            label="Add document"
            icon="SquarePlus"
            onClick={() => fileUploadRef.current?.click()}
          />
          {documents.length > 0 ? (
            <>
              <MenuItem
                label="Chat with claim"
                icon="WandMagicSparkles"
                linkTo={`/claims/${uniqueID}/chats`}></MenuItem>
              <ExportTableMenuItem onClick={() => tableRef.current?.handleExport()} />
              <FilterTableMenuItem
                filterTerms={filterTerms}
                onChange={(terms) => setFilterTerms(terms)}
              />
            </>
          ) : (
            <></>
          )}
        </Menu>
        <input
          type="file"
          accept=".pdf"
          ref={fileUploadRef}
          onChange={handleFileChange}
          style={{ visibility: 'hidden', width: '0' }}
        />
        <Table
          ref={tableRef}
          id="claim-documents-table"
          name="Claim documents"
          keyField="uniqueID"
          columns={columns}
          sourceData={documents}
          isPropertyBarVisible={false}
          filterTerms={filterTerms}
          onDelete={confirmDelete}
          onRowClick={handleRowClick}
          initialSortColumn={'name'}></Table>
        <DocumentViewer
          theDocument={clickedDocument}
          claimUniqueID={uniqueID}
          citations={undefined}
          onCancel={() => {
            setClickedDocument(undefined)
          }}></DocumentViewer>
      </>
    )
  )
}

export default ClaimDocuments
