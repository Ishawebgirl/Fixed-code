import { useEffect, useMemo, useRef, useState } from 'react'
import Table from 'components/Table'
import * as Model from 'api/model'
import * as signalR from '@microsoft/signalr'
import moment from 'moment'
import FilterTableMenuItem from 'components/FilterTableMenuItem'
import ExportTableMenuItem from 'components/ExportTableMenuItem'
import Menu from 'components/Menu'
import { Log } from 'helpers/log'

const Jobs = () => {
  const jobApi = useMemo(() => new Model.JobClient(process.env.REACT_APP_API_URL), [])

  const tableRef = useRef<{ handleExport: () => void } | null>(null)

  const [jobs, setJobs] = useState<Model.Job[]>()
  const [filterTerms, setFilterTerms] = useState('')

  const columns = useMemo(
    () => [
      {
        label: 'Description',
        accessor: 'description',
      },
      {
        label: 'Status',
        accessor: 'status',
        type: 'propertyTag',
      },
      {
        label: 'Interval',
        accessor: 'interval',
        type: 'interval',
      },
      {
        label: 'Timeout',
        accessor: 'timeout',
        type: 'interval',
      },
      {
        label: 'Next run',
        accessor: 'nextEvent',
        type: 'timeAgo',
      },
    ],
    []
  )

  const hubConnection = new signalR.HubConnectionBuilder()
    .withUrl(process.env.REACT_APP_SIGNALR_URL + '/hub/job', {
      transport: signalR.HttpTransportType.WebSockets,
      skipNegotiation: true,
    })
    .configureLogging(signalR.LogLevel.Error)
    .withAutomaticReconnect()
    .build()

  hubConnection.start().catch((err) => Log.add(err))

  hubConnection.onreconnected((connectionId) => {
    //Log.add('signalr reconnected')
  })

  hubConnection.onreconnecting((error) => {
    //Log.add('signalr reconnecting', error)
  })

  hubConnection.onclose((e) => {
    //Log.add('signalr connection closed', e)
  })

  hubConnection.on('SetJobStatus', (id: number, status: string, nextEvent: string) => {
    const job = jobs?.find((x) => x.id === id)

    if (job) {
      nextEvent += 'Z' // booooo, can't add custom datetime formatter to signalr init on the backend :("
      Log.add(
        `Updating jobStatus for ID ${id} to ${status} and nextEvent to ${nextEvent}`
      )

      setTimeout(
        () => {
          job.status = status as Model.JobStatus
          job.nextEvent = moment(nextEvent)
          setJobs([...jobs!])
        },
        status === 'Pending' ? 2000 : 0
      )
    }
  })

  useEffect(() => {
    ;(async () => {
      try {
        const result = await jobApi.getAllJobs()
        setJobs(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [jobApi])

  return (
    jobs && (
      <>
        <div className="header">Scheduled Jobs</div>
        <Menu>
          {jobs.length > 0 ? (
            <>
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
        <Table
          ref={tableRef}
          id="job-table"
          name="Jobs"
          keyField="id"
          columns={columns}
          sourceData={jobs}
          isPropertyBarVisible={false}
          onRowClick={null}
          filterTerms={filterTerms}
        />
      </>
    )
  )
}

export default Jobs
