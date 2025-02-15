import { useEffect, useState } from 'react'
import USAMap from 'react-usa-map'

interface IMap {
  data: { [key: string]: number } | undefined
}

const colors = ['#0f75bc', '#598bc4', '#7e9fcc', '#a4b8d8', '#ced8e8']

const Map = ({ data }: IMap) => {
  const [stateData, setStateData] = useState<
    { [key: string]: { fill: string; group: number } } | undefined
  >()

  const mapHandler = (event: { target: { dataset: { name: any } } }) => {
    // alert(event.target.dataset.name)
  }

  useEffect(() => {
    if (data === undefined) {
      return
    }

    const maxCount = Math.max(...Object.values(data!))

    if (maxCount === 0) {
      return
    }

    const groupSize = maxCount / colors.length
    const mappedData = Object.entries(data!).reduce(
      (
        acc: { [key: string]: { fill: string; group: number; count: number } },
        [state, count]: [string, number]
      ) => {
        const group = colors.length - Math.ceil(count / groupSize)
        acc[state] = { fill: colors[group], group, count }
        return acc
      },
      {}
    )

    setStateData(mappedData)
  }, [data])

  return (
    <div className="map">
      <USAMap
        title={''}
        defaultFill={'#f0f0f0'}
        customize={stateData}
        onClick={mapHandler}
      />
    </div>
  )
}

export default Map
