import * as Model from 'api/model'
import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import moment from 'moment'

interface IStackedBarChart {
  data: { [key: string]: Model.ClaimStatusValue[] } | undefined
}

const colors = ['#0f75bc', '#598bc4', '#7e9fcc', '#a4b8d8', '#ced8e8']
const statuses = ['Unassigned', 'Investigating', 'Adjudicated', 'Resolved']

const StackedBarChart = ({ data }: IStackedBarChart) => {
  const [mappedData, setMappedData] = useState<any[]>([])

  useEffect(() => {
    if (data === undefined) {
      return
    }

    let dict = {} as any
    const now = moment().utc()
    const yearAgo = moment({ year: now.year(), month: now.month(), day: 1 }).add(
      -13,
      'month'
    )

    for (let month = 0; month < 12; month++) {
      const key = yearAgo.add(1, 'month').format('YYYY-MM-DD') + 'T00:00:00'

      dict[key] = {
        name: key,
        Unassigned: 0,
        Investigating: 0,
        Adjudicated: 0,
        Resolved: 0,
      }
    }

    for (const [key, value] of Object.entries(data)) {
      for (const claim of value) {
        dict[key][claim.status] = claim.value
      }
    }
    setMappedData(Object.values(dict))
  }, [data])

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        style={{ opacity: mappedData.length === 0 ? 0 : 1 }}
        data={mappedData}
        margin={{
          top: 10,
          right: 0,
          left: 0,
          bottom: 0,
        }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          padding={{ left: 8 }}
          tickFormatter={(tick) => {
            const mmm = moment(tick).format('MMM')
            return mmm + (mmm === 'Jan' ? moment(tick).format(' YYYY') : '')
          }}
        />
        <YAxis
          padding={{ top: 0 }}
          tickFormatter={(tick) => {
            return `$${tick / 1000}k`
          }}
        />
        {statuses.map((status, index) => (
          <Bar
            key={index}
            dataKey={status}
            stackId="a"
            isAnimationActive={false}
            fill={colors[statuses.length - 1 - index]}
            radius={0}>
            {mappedData.map((entry, index) => {
              const keys = Object.keys(entry)
              const values = Object.values(entry)

              const statusIndex = keys.findIndex((key) => key === status)
              const lastBarIndex = values.findLastIndex((value) => value !== 0)

              return (
                <Cell
                  key={`cell-${index}`}
                  radius={statusIndex === lastBarIndex ? [5, 5, 0, 0] : (0 as any)}
                />
              )
            })}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

export default StackedBarChart
